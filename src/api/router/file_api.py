
"""File operations API routes.

Provides REST API endpoints for file upload, download, update, and deletion.
Supports space- and folder-based file management.
"""
import traceback
from urllib.parse import quote
from uuid import UUID
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Depends, Request
from fastapi.responses import StreamingResponse
from typing import Optional, List
from sqlalchemy.exc import IntegrityError, NoResultFound
from pydantic import ValidationError
from fastmcp.exceptions import ToolError

from src.auth.dependencies import authenticate_request
from src.api.dependencies.auth import extract_token
from src.api.router.response import (
    BatchDeleteRequest,
    ErrorDetailResponse,
    FileDeletedResponse,
    FileResource,
    FileUploadResponse,
    UpdateFileRequest,
)
from src.adapter.file import FileAdapter
from src.adapter.folder import FolderAdapter
from src.adapter.rag import get_rag_adapter
from src.log import get_api_logger
from db.file_with_index_db import FileWithIndexDB

logger = get_api_logger()

router = APIRouter(tags=["Files"], prefix="/folders", dependencies=[Depends(authenticate_request)])

# When auto_index=True the indexing block includes a background job_id; otherwise enabled=false
@router.post(
    path="/{folder_id}/file",
    summary="Upload File",
    description="Upload a file to a specified folder",
    response_model=FileUploadResponse,
    responses={
        404: {"model": ErrorDetailResponse, "description": "folder not found"},
        409: {"model": ErrorDetailResponse, "description": "a file with the same name already exists in the folder"},
        400: {"model": ErrorDetailResponse, "description": "validation failed (file too large, unsupported format, etc.)"},
        500: {"model": ErrorDetailResponse},
    },
)
async def upload_file(
    folder_id: int,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    auto_index: bool = Form(True),
    user_token: str = Depends(extract_token),
):
    """
    Args:
        folder_id: Folder ID
        file: The file to upload
        description: File description (optional)
        tags: File tags (optional; comma-separated string, e.g. "tag1,tag2,tag3")
        auto_index: Whether to automatically index the file into the vector store; defaults to True
    """

    logger.info(f"Starting file upload: {file.filename} to folder {folder_id}, auto_index={auto_index})")
    logger.info(f"Received description: {description}, tags: {tags}")

    try:
        result = await FolderAdapter.get_folder(id=folder_id, user_token=user_token)
        if not result:
            raise HTTPException(status_code=404, detail=f"folder {folder_id} not found")
        file_content = await file.read()

        tags_list = None
        if tags:
            tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()]

        file_data = {
            'file_content': file_content,
            'file_name': file.filename,
            'description': description,
            'tags': tags_list
        }

        upload_result = await FileAdapter.upload_file(
            files_data=file_data,
            folder_id=folder_id,
            user_token=user_token
        )

        logger.info(f"File upload completed: {file.filename} to folder {folder_id}")

        indexing_metadata = None
        if auto_index:
            try:
                if isinstance(upload_result, list):
                    uploaded_file_ids = [str(f.id) for f in upload_result]
                else:
                    # This shouldn't happen for single file, but handle gracefully
                    uploaded_file_ids = []

                if uploaded_file_ids:
                    indexing_metadata = await get_rag_adapter().trigger_auto_index(
                        file_ids=uploaded_file_ids,
                        folder_id=folder_id,
                        token=user_token,
                        auto_index=True
                    )
                    logger.info(f"Auto-indexing triggered for uploaded file: {file.filename}")
            except Exception as e:
                # Log warning but don't fail - file upload succeeded
                logger.warning(f"Auto-indexing failed for uploaded file {file.filename}: {str(e)}")
                indexing_metadata = {
                    "auto_index_enabled": True,
                    "message": f"Warning: Auto-indexing failed: {str(e)}"
                }

        if isinstance(upload_result, list) and len(upload_result) > 0:
            file_info = upload_result[0].model_dump() if hasattr(upload_result[0], 'model_dump') else upload_result[0]
            response = {
                "file_info": file_info,
                "indexing": indexing_metadata if indexing_metadata else {"auto_index_enabled": False, "message": "Auto-indexing disabled"},
                "message": "File uploaded successfully"
            }
        else:
            file_info = upload_result.model_dump() if hasattr(upload_result, 'model_dump') else upload_result
            response = {
                "file_info": file_info,
                "indexing": indexing_metadata if indexing_metadata else {"auto_index_enabled": False, "message": "Auto-indexing disabled"},
                "message": "File uploaded successfully"
            }

        return response

    except IntegrityError as e:
        traceback.print_exc()
        logger.error(f"Database integrity error uploading file {file.filename if file else 'unknown'} to folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=409, detail="file already exists")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to upload file {file.filename if file else 'unknown'} to folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get(
    path="/{folder_id}/files/{file_id}/download",
    summary="Download File",
    description="Download a specified file",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"application/octet-stream": {}},
            "description": "file content binary stream (Content-Type depends on the file's mime_type)",
        },
        404: {"model": ErrorDetailResponse, "description": "folder / file not found"},
        500: {"model": ErrorDetailResponse},
    },
)
async def download_file(
    folder_id: int,
    file_id: UUID,
    user_token: str = Depends(extract_token),
):
    """
    Args:
        folder_id: Folder ID
        file_id: File ID (UUID)
    """
    try:
        result = await FolderAdapter.get_folder(id=folder_id, user_token=user_token)
        if not result:
            raise HTTPException(status_code=404, detail=f"folder {folder_id} not found")
        
        file_info_result = await FileAdapter.get_file(user_token=user_token, folder_id=folder_id, id=file_id)

        if not file_info_result:
            logger.warning(f"File not found: file_id={file_id}, folder_id={folder_id}")
            raise HTTPException(status_code=404, detail="File not found")

        file_info = file_info_result[0]

        result = await FileAdapter.download_file(
            folder_id=file_info.folder_id,
            file_id=file_id,
            user_token=user_token,
        )

        if not result:
            logger.warning(f"Failed to download file: file_id={file_id}, folder_id={file_info.folder_id}")
            raise HTTPException(status_code=404, detail="File not found")

        file_data = result[0]
        logger.info(f"File downloaded successfully: file_id={file_id}, filename={file_data.file_name}")
        ascii_filename = file_data.file_name.encode('ascii', 'ignore').decode('ascii') or 'download'
        encoded_filename = quote(file_data.file_name)

        def generate():
            yield file_data.file_content

        return StreamingResponse(
            generate(),
            media_type=file_data.mime_type or "application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{ascii_filename}"; filename*=UTF-8\'\'{encoded_filename}'
            }
        )
    except ToolError as e:
        traceback.print_exc()
        logger.error(f"Tool error downloading file {file_id} in folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except NoResultFound as e:
        traceback.print_exc()
        logger.warning(f"File not found: file_id={file_id}, folder_id={folder_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        traceback.print_exc()
        logger.error(f"Validation error downloading file {file_id} in folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Data validation failed: {str(e)}")
    except ValueError as e:
        traceback.print_exc()
        logger.error(f"Value error downloading file {file_id} in folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to download file {file_id} in folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.patch(
    "/{folder_id}/files/{file_id}",
    summary="Update file metadata",
    response_model=FileResource,
    responses={
        404: {"model": ErrorDetailResponse, "description": "folder / file not found"},
        409: {"model": ErrorDetailResponse, "description": "rename collides with an existing name"},
        400: {"model": ErrorDetailResponse, "description": "invalid parameters"},
    },
)
async def update_file(
    folder_id: int,
    file_id: UUID,
    requests: UpdateFileRequest,
    user_token: str = Depends(extract_token),
):
    """Update the metadata of a specified file."""
    try:
        result = await FolderAdapter.get_folder(id=folder_id, user_token=user_token)
        if not result:
            raise HTTPException(status_code=404, detail=f"folder {folder_id} not found")
        
        result = await FileAdapter.update_file_metadata(
            folder_id=folder_id,
            file_id=file_id,
            user_token=user_token,
            description=requests.description,
            tags=requests.tags
        )

        logger.info(f"File metadata updated successfully: file_id={file_id}, folder_id={folder_id}")
        # response_model=FileResource (single) -> return result[0] directly; do not wrap it in
        # `{"file_info": ...}` (that shape belongs to upload_file's FileUploadResponse).
        return result[0]

    except ToolError as e:
        traceback.print_exc()
        logger.error(f"Tool error updating file metadata {file_id} in folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except NoResultFound as e:
        traceback.print_exc()
        logger.error(f"File not found for update: file_id={file_id}, folder_id={folder_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        traceback.print_exc()
        logger.error(f"Validation error updating file metadata {file_id} in folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Data validation failed: {str(e)}")
    except ValueError as e:
        traceback.print_exc()
        logger.error(f"Value error updating file metadata {file_id} in folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to update file metadata {file_id} in folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Update file metadata failed: {str(e)}")


@router.delete(
    "/{folder_id}/files/{file_id}",
    summary="Delete file",
    response_model=FileDeletedResponse,
    responses={
        404: {"model": ErrorDetailResponse, "description": "folder / file not found"},
        500: {"model": ErrorDetailResponse},
    },
)
async def delete_file(
    folder_id: int,
    file_id: UUID,
    user_token: str = Depends(extract_token),
):
    """Delete a specified file."""
    try:
        result = await FolderAdapter.get_folder(id=folder_id, user_token=user_token)
        if not result:
            raise HTTPException(status_code=404, detail=f"folder {folder_id} not found")
        
        file_info_result = await FileAdapter.get_file(user_token=user_token, folder_id=folder_id, id=file_id)

        if not file_info_result:
            logger.warning(f"File not found for deletion: file_id={file_id}, folder_id={folder_id}")
            raise NoResultFound("File not found")

        file_info = file_info_result[0]

        result = await FileAdapter.delete_file(
            file_ids=file_id,
            folder_id=file_info.folder_id,
            user_token=user_token
        )

        logger.info(f"File deleted successfully: file_id={file_id}, folder_id={folder_id}")
        return {"success": result, "message": "File deleted successfully"}

    except ToolError as e:
        traceback.print_exc()
        logger.error(f"Tool error deleting file {file_id} in folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except NoResultFound as e:
        traceback.print_exc()
        logger.error(f"File not found for deletion: file_id={file_id}, folder_id={folder_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        traceback.print_exc()
        logger.error(f"Validation error deleting file {file_id} in folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Data validation failed: {str(e)}")
    except ValueError as e:
        traceback.print_exc()
        logger.error(f"Value error deleting file {file_id} in folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to delete file {file_id} in folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete file failed: {str(e)}")


# Batch upload — the response shape differs from single-file: it includes successful/failed/skipped lists
# (the adapter already returns a dict; no response_model is enforced here because the nested batch structure is complex)
@router.post(
    "/{folder_id}/files",
    summary="Upload multiple files (batch)",
    responses={
        404: {"model": ErrorDetailResponse, "description": "folder not found"},
        409: {"model": ErrorDetailResponse, "description": "some files have name conflicts"},
        500: {"model": ErrorDetailResponse},
    },
)
async def upload_files(
    folder_id: int,
    files: List[UploadFile] = File(...),
    auto_index: bool = Form(True),
    user_token: str = Depends(extract_token),
):
    """Batch-upload files to a specified folder.

    Args:
        folder_id: Folder ID
        files: The list of files to upload
        auto_index: Whether to automatically index files into the vector store; defaults to True
    """
    try:
        result = await FolderAdapter.get_folder(id=folder_id, user_token=user_token)
        if not result:
            raise HTTPException(status_code=404, detail=f"folder {folder_id} not found")

        files_data = []
        for file in files:
            file_content = await file.read()
            files_data.append({
                'file_content': file_content,
                'file_name': file.filename,
                'description': None,
                'tags': None
            })

        logger.info(f"Starting batch upload: {len(files)} files to folder {folder_id} (auto_index={auto_index})")

        upload_result = await FileAdapter.upload_file(
            files_data=files_data,
            folder_id=folder_id,
            user_token=user_token
        )

        logger.info(f"Batch upload completed: folder={folder_id}")

        indexing_metadata = None
        if auto_index:
            try:
                uploaded_file_ids = []
                if isinstance(upload_result, dict):
                    if 'successful_uploads' in upload_result:
                        uploaded_file_ids = [str(f['id']) for f in upload_result['successful_uploads']]
                    elif 'files' in upload_result:
                        uploaded_file_ids = [str(f['id']) for f in upload_result['files']]

                if uploaded_file_ids:
                    indexing_metadata = await get_rag_adapter().trigger_auto_index(
                        file_ids=uploaded_file_ids,
                        folder_id=folder_id,
                        token=user_token,
                        auto_index=True
                    )
                    logger.info(f"Auto-indexing triggered for {len(uploaded_file_ids)} uploaded files")
                else:
                    indexing_metadata = {
                        "auto_index_enabled": False,
                        "message": "No files were successfully uploaded to index"
                    }
            except Exception as e:
                # Log warning but don't fail - file upload succeeded
                logger.warning(f"Auto-indexing failed for batch upload: {str(e)}")
                indexing_metadata = {
                    "auto_index_enabled": True,
                    "message": f"Warning: Auto-indexing failed: {str(e)}"
                }

        response = {
            **upload_result,
            "indexing": indexing_metadata if indexing_metadata else {"auto_index_enabled": False, "message": "Auto-indexing disabled"}
        }

        return response

    except ToolError as e:
        traceback.print_exc()
        logger.error(f"Tool error in batch upload to folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except NoResultFound as e:
        traceback.print_exc()
        logger.error(f"Folder not found for batch upload: folder_id={folder_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except IntegrityError as e:
        traceback.print_exc()
        logger.error(f"Database integrity error in batch upload to folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=409, detail="File already exists or data conflict")
    except ValidationError as e:
        traceback.print_exc()
        logger.error(f"Validation error in batch upload to folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Data validation failed: {str(e)}")
    except ValueError as e:
        traceback.print_exc()
        logger.error(f"Value error in batch upload to folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed batch upload to folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")


# When include_index_status=True each entry also carries indexed_at / status / num_chunks (via JOIN)
@router.get(
    "/{folder_id}/files",
    summary="List or get files",
    responses={
        404: {"model": ErrorDetailResponse, "description": "folder / file not found"},
        400: {"model": ErrorDetailResponse, "description": "invalid parameters"},
    },
)
async def list_files(
    folder_id: int,
    file_id: Optional[UUID] = Query(None, description="file ID (if provided, returns info for the single file)"),
    file_name: Optional[str] = Query(None, description="file name (if provided, looks up the file by name)"),
    search_tags: Optional[str] = Query(None, description="search tags (partial match)"),
    mime_type_filter: Optional[str] = Query(None, description="MIME type filter (partial match)"),
    include_index_status: bool = Query(True, description="include file index status (JOIN-optimized)"),
    user_token: str = Depends(extract_token),
):
    """List files in a folder, or fetch a single one by file_id / file_name.

    Uses a JOIN query to fetch file + index status in one shot, avoiding N+1 (10-100x faster on
    large folders). If index info isn't needed, set include_index_status=False for extra speed.

    Args:
        folder_id: Folder id.
        file_id: If given, returns a single file; highest priority.
        file_name: If given, looks up by name; second priority.
        search_tags: Partial tag match.
        mime_type_filter: Partial MIME match.
        include_index_status: True -> return fields including indexed_at / status / num_chunks.
        user_token: User token (permission check).

    Returns:
        A list or dict (depending on the query parameters); each entry has the basic File fields
        plus optional index fields.
    """
    try:
        folder_list = await FolderAdapter.get_folder(id=folder_id, user_token=user_token)
        if not folder_list:
            raise HTTPException(status_code=404, detail=f"folder {folder_id} not found")
        
        folder = folder_list[0]

        if file_id is not None:
            logger.info(f"Fetching file info: file_id={file_id} in folder {folder_id}")
            result = await FileAdapter.get_file(user_token=user_token, folder_id=folder.id, id=file_id)
            
            if not result:
                raise NoResultFound("File not found")
            
            logger.info(f"File info retrieved successfully: file_id={file_id}, folder_id={folder_id}")
            return {"file_info": result[0]}

        if file_name:
            logger.info(f"Fetching file info by name: file_name={file_name} in folder {folder_id}")
            result = await FileAdapter.get_file(user_token=user_token, folder_id=folder.id, file_name=file_name)
            
            if not result:
                raise NoResultFound(f"File with name '{file_name}' not found")
            
            logger.info(f"File info retrieved successfully by name: file_name={file_name}, folder_id={folder_id}")
            return {"file_info": result[0]}

        if include_index_status:
            files_with_index = FileWithIndexDB.get_files_with_index_status(folder.id)

            if not files_with_index:
                logger.info(f"No files found in folder: folder_id={folder_id}")
                return {"files": [], "message": "No files found in folder"}

            filtered_files = files_with_index
            if search_tags:
                # Search in JSONB array: check if any tag contains the search string
                filtered_files = [
                    f for f in filtered_files
                    if f.get('tags') and isinstance(f['tags'], list) and 
                    any(search_tags.lower() in tag.lower() for tag in f['tags'])
                ]
            if mime_type_filter:
                filtered_files = [
                    f for f in filtered_files
                    if f.get('mime_type') and mime_type_filter.lower() in f['mime_type'].lower()
                ]

            logger.info(f"Files listed successfully with index status: folder_id={folder_id}, count={len(filtered_files)}")
            return {"files": filtered_files}

        else:
            # Legacy behavior: fetch files only (no index status)
            result = await FileAdapter.get_file(user_token=user_token, folder_id=folder.id)

            if not result:
                logger.info(f"No files found in folder: folder_id={folder_id}")
                return {"files": [], "message": "No files found in folder"}

            filtered_files = result
            if search_tags:
                # Search in JSONB array: check if any tag contains the search string
                filtered_files = [
                    f for f in filtered_files 
                    if f.tags and isinstance(f.tags, list) and 
                    any(search_tags.lower() in tag.lower() for tag in f.tags)
                ]
            if mime_type_filter:
                filtered_files = [f for f in filtered_files if f.mime_type and mime_type_filter.lower() in f.mime_type.lower()]

            logger.info(f"Files listed successfully: folder_id={folder_id}, count={len(filtered_files)}")
            return {"files": filtered_files}

    except ToolError as e:
        traceback.print_exc()
        logger.error(f"Tool error listing files in folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except NoResultFound as e:
        traceback.print_exc()
        logger.error(f"Folder not found for listing files: folder_id={folder_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        traceback.print_exc()
        logger.error(f"Validation error listing files in folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Data validation failed: {str(e)}")
    except ValueError as e:
        traceback.print_exc()
        logger.error(f"Value error listing files in folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to list files in folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"List files failed: {str(e)}")


@router.delete(
    "/{folder_id}/files",
    summary="Delete multiple files (batch)",
    responses={
        404: {"model": ErrorDetailResponse, "description": "folder not found"},
        500: {"model": ErrorDetailResponse},
    },
)
async def delete_files(
    requests: BatchDeleteRequest,
    folder_id: int,
    user_token: str = Depends(extract_token),
):
    """Batch-delete files in a specified folder."""
    try:
        folder_list = await FolderAdapter.get_folder(id=folder_id, user_token=user_token)
        if not folder_list:
            raise HTTPException(status_code=404, detail=f"folder {folder_id} not found")
        
        folder = folder_list[0]

        result = await FileAdapter.delete_file(
            file_ids=requests.file_ids,
            folder_id=folder.id,
            user_token=user_token,
            delete_all=requests.delete_all
        )

        logger.info(f"Batch file deletion completed: folder_id={folder_id}, delete_all={requests.delete_all}")
        return result

    except ToolError as e:
        traceback.print_exc()
        logger.error(f"Tool error in batch file deletion for folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except NoResultFound as e:
        traceback.print_exc()
        logger.error(f"Folder not found for batch deletion: folder_id={folder_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        traceback.print_exc()
        logger.error(f"Validation error in batch file deletion for folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Data validation failed: {str(e)}")
    except ValueError as e:
        traceback.print_exc()
        logger.error(f"Value error in batch file deletion for folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed batch file deletion for folder {folder_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch delete files failed: {str(e)}")
