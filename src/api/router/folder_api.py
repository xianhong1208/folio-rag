
"""Folder operations API routes.

Provides REST API endpoints for folder creation, querying, updating, and deletion.
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Query
import traceback
from sqlalchemy.exc import IntegrityError, NoResultFound
from pydantic import ValidationError
from typing import Optional
from typing import List, Union

from src.api.router.response import (
    CreateFolderRequest,
    ErrorDetailResponse,
    FolderDeletedResponse,
    FolderResource,
    UpdateFolderRequest,
)
from src.adapter.folder import FolderAdapter
from src.log import get_api_logger
from src.auth.dependencies import authenticate_request
from src.api.dependencies.auth import extract_token
logger = get_api_logger()

router = APIRouter(tags=["Folders"], prefix="/folders", dependencies=[Depends(authenticate_request)])

# Resource = FolderResource. Response = List[FolderResource] (even a by-ID lookup is wrapped in a list, preserving the adapter's existing behavior).
@router.get(
    "/",
    summary="List or get folders",
    response_model=List[FolderResource],
    responses={
        404: {"model": ErrorDetailResponse, "description": "folder_id / folder_name not found"},
        400: {"model": ErrorDetailResponse, "description": "invalid parameter type"},
    },
)
async def list_folders(
    folder_id: Optional[int] = Query(None, description="folder ID"),
    folder_name: Optional[str] = Query(None, description="folder name"),
    user_token: str = Depends(extract_token),
):
    """List folders, or fetch a single one by ID / name.

    Args:
        folder_id: If given, returns that ID; highest priority.
        folder_name: If given, looks up by name; second priority.
        user_token: User token (permission check).

    Returns:
        A FolderResource list (even a by-ID lookup is wrapped in a list, preserving the adapter's behavior).
    """
    try:
        logger.debug(f'user_token: {user_token[:8] if user_token else "N/A"}...')

        if folder_id is not None:
            result = await FolderAdapter.get_folder(id=folder_id, user_token=user_token)
            if not result:
                raise HTTPException(status_code=404, detail=f"folder {folder_id} not found")
            return result

        if folder_name:
            result = await FolderAdapter.get_folder(name=folder_name, user_token=user_token)
            if not result:
                raise HTTPException(status_code=404, detail=f"folder with name '{folder_name}' not found")
            return result

        result = await FolderAdapter.get_folder(user_token=user_token)
        return result
    except NoResultFound as e:
        traceback.print_exc()
        logger.info(f"No folders found")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        traceback.print_exc()
        logger.error(f"Invalid folder ID or parameters: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        # Re-raise HTTPExceptions as-is to preserve their status codes
        raise e
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to list/get folders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/",
    summary="Create folder",
    response_model=List[FolderResource],
    responses={
        409: {"model": ErrorDetailResponse, "description": "a folder with the same name already exists"},
        400: {"model": ErrorDetailResponse, "description": "invalid parameters"},
        422: {"model": ErrorDetailResponse, "description": "Pydantic validation failed"},
    },
)
async def create_folder(
    folder_request: CreateFolderRequest,
    user_token: str = Depends(extract_token),
):
    """Create a new folder (bound to the current user_token).

    Args:
        folder_request: CreateFolderRequest (name + optional description).
        user_token: User token (stored in Folder.user_token, used later for permission filtering).

    Returns:
        The newly created FolderResource.
    """
    try:
        logger.debug(f'user_token: {user_token[:8] if user_token else "N/A"}...')

        result = FolderAdapter.create_folder(
            name=folder_request.name,
            user_token=user_token,
            description=folder_request.description
        )

        return result
        
    except IntegrityError as e:
        traceback.print_exc()
        logger.error(f"Data integrity error while creating folder: {str(e)}")
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        traceback.print_exc()
        logger.error(f"Value error while creating folder: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError as e:
        traceback.print_exc()
        logger.error(f"Pydantic validation error creating folder: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except NoResultFound as e:
        traceback.print_exc()
        logger.error(f"Folder not found while creating folder: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException as e:
        # Re-raise HTTPExceptions as-is to preserve their status codes
        raise e
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to create folder: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Response = {detail: "Folder X deleted successfully"} (deleting a folder also cancels in-progress indexing jobs).
@router.delete(
    "/{folder_id}",
    summary="delete folder",
    response_model=FolderDeletedResponse,
    responses={
        404: {"model": ErrorDetailResponse, "description": "folder not found or no permission"},
        400: {"model": ErrorDetailResponse, "description": "invalid parameters"},
    },
)
async def delete_folder(
    folder_id: int,
    user_token: str = Depends(extract_token),
):
    """Delete the folder with the given ID.

    Deleting a folder automatically deletes all files within it (soft delete).
    Any still-running indexing job is cancelled first, so it doesn't keep writing to a folder that is about to disappear.

    Args:
        folder_id: Folder ID
    """
    try:
        logger.debug(f'user_token: {user_token[:8] if user_token else "N/A"}...')

        # Verify ownership before any side effect. cancel_jobs_for_folder takes only folder_id,
        # not a token; placed before the ownership check, any tenant DELETEing someone else's
        # folder_id would kill the other tenant's in-flight job before getting a 404 (cross-tenant
        # DoS). Fail-closed: unowned (user_token=None) or not-the-owner always returns 404, without
        # leaking whether the folder exists.
        from db.cached_folderdb import CachedFolderDB
        _folder = CachedFolderDB.get_by_id(folder_id)
        if not _folder or _folder.user_token != user_token:
            raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")

        # cancel any in-flight indexing jobs for this folder first
        try:
            from src.domain.rag.index_job_manager import IndexingJobManager
            cancelled = await IndexingJobManager.get_instance().cancel_jobs_for_folder(folder_id)
            if cancelled:
                logger.info(f"Cancelled {len(cancelled)} in-flight job(s) before deleting folder {folder_id}")
        except Exception as cancel_err:
            logger.warning(f"Pre-delete job cancellation failed (continuing): {cancel_err}")

        result = await FolderAdapter.delete_folder(folder_id=folder_id, user_token=user_token)
        if result:
            return {"detail": f"Folder {folder_id} deleted successfully"}

    except NoResultFound as e:
        traceback.print_exc()
        logger.info(f"Folder {folder_id} not found for deletion")
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException as e:
        raise e
    except ValueError as e:
        traceback.print_exc()
        logger.error(f"Value error deleting folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to delete folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Resource = FolderResource (after update) — the adapter returns a list wrapper (preserving
# historical behavior), so response_model uses List, consistent with list_folders / create_folder.
@router.patch(
    "/{folder_id}",
    summary="Update folder details",
    response_model=List[FolderResource],
    responses={
        404: {"model": ErrorDetailResponse, "description": "folder not found"},
        409: {"model": ErrorDetailResponse, "description": "rename collides with an existing name"},
        400: {"model": ErrorDetailResponse, "description": "invalid parameters"},
    },
)
async def update_folder_api(
    folder_id: int,
    requests: UpdateFolderRequest,
    user_token: str = Depends(extract_token),
):
    """Update a folder's name / description (only patches the fields provided).

    Args:
        folder_id: The folder to update.
        requests: UpdateFolderRequest (name / description both optional).
        user_token: User token (permission check).

    Returns:
        The updated FolderResource.
    """
    try:
        logger.debug(f'user_token: {user_token[:8] if user_token else "N/A"}...')

        result = await FolderAdapter.update_folder(
            folder_id=folder_id,
            user_token=user_token,
            name=requests.name,
            description=requests.description
        )
        
        return result
        
    except NoResultFound:
        traceback.print_exc()
        logger.info(f"Folder {folder_id} not found for update")
        raise HTTPException(status_code=404)
    except IntegrityError as e:
        traceback.print_exc()
        logger.error(f"Data integrity error updating folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        traceback.print_exc()
        logger.error(f"Value error updating folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError as e:
        traceback.print_exc()
        logger.error(f"Pydantic validation error updating folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to update folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))