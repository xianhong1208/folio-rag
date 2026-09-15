"""Generic base class providing CRUD operations for ORM-backed tables."""

from db.db import Session as DBSession
from sqlalchemy.exc import IntegrityError

# Sensitive fields to mask in logs
SENSITIVE_FIELDS = {'user_token', 'token', 'password', 'secret'}


def _mask_sensitive_data(data: dict) -> dict:
    """Mask sensitive fields, keeping only the first 8 characters."""
    masked = {}
    for key, value in data.items():
        if key in SENSITIVE_FIELDS and isinstance(value, str) and len(value) > 8:
            masked[key] = f"{value[:8]}..."
        else:
            masked[key] = value
    return masked


class BaseDB:
    """Base class providing generic CRUD operations for a subclass's ORM table."""


    @classmethod
    def get_orm_class(cls):
        raise NotImplementedError("Subclasses must implement get_orm_class method")

    @classmethod
    def get_log(cls):
        raise NotImplementedError("Subclasses must implement get_log method")

    @classmethod
    def _get_valid_fields(cls, query_kwargs) -> dict:
        valid_fields = list(cls.get_orm_class().__table__.columns.keys())
        invalid_fields = [k for k in query_kwargs.keys() if k not in valid_fields]
        if invalid_fields:
            cls.get_log().error(f"Invalid field names: {invalid_fields}. Valid fields are: {valid_fields}")
            raise ValueError(f"Invalid field names: {invalid_fields}")
        return query_kwargs
    
    @classmethod
    def get(cls, **kwargs):
        try:
            query_kwargs = cls._get_valid_fields(kwargs)
            cls.get_log().info(f"Getting {cls.__name__} with filters: {_mask_sensitive_data(query_kwargs)}")
            with DBSession() as session:
                result = session.query(cls.get_orm_class()).filter_by(**query_kwargs).all()
                return result
        except ValueError:
            raise
        except Exception as e:
            cls.get_log().error(f"Failed to get {cls.__name__}: {e}")
            raise

    @classmethod
    def create(cls, **kwargs):
        with DBSession() as session:
            try:
                cls.get_log().info(f"Creating {cls.__name__} with: {_mask_sensitive_data(kwargs)}")
                obj = cls.get_orm_class()(**kwargs)
                session.add(obj)
                session.commit()
                session.refresh(obj)
                cls.get_log().info(f"Successfully created {cls.__name__}: {obj}")
                return obj
            except IntegrityError as e:
                session.rollback()
                cls.get_log().error(f"Integrity constraint violation for {cls.__name__}: {e}")
                raise
            except Exception as e:
                session.rollback()
                cls.get_log().error(f"Failed to create {cls.__name__}: {e}")
                raise

    @classmethod
    def update(cls, id, **kwargs):
        with DBSession() as session:
            try:
                obj = session.query(cls.get_orm_class()).filter_by(id=id).first()
                if not obj:
                    raise ValueError(f"{cls.__name__} id {id} does not exist")
                for k, v in kwargs.items():
                    if v is not None and hasattr(obj, k):
                        setattr(obj, k, v)
                session.commit()
                session.refresh(obj)
                session.expunge(obj)
                cls.get_log().info(f"Successfully updated {cls.__name__}: {obj}")
                return obj
            except ValueError:
                raise
            except IntegrityError as e:
                session.rollback()
                cls.get_log().error(f"Integrity constraint violation for {cls.__name__}: {e}")
                raise
            except Exception as e:
                session.rollback()
                cls.get_log().error(f"Failed to update {cls.__name__}: {e}")
                raise

    @classmethod
    def delete(cls, **kwargs):
        try:
            with DBSession() as session:
                obj = session.query(cls.get_orm_class()).filter_by(**kwargs).first()
                if obj:
                    session.delete(obj)
                    session.commit()
                    cls.get_log().info(f"Successfully deleted {cls.__name__}: {_mask_sensitive_data(kwargs)}")
                    return True
                else:
                    cls.get_log().warning(f"{cls.__name__} with {_mask_sensitive_data(kwargs)} not found")
                    return False
        except Exception as e:
            cls.get_log().error(f"Failed to delete {cls.__name__}: {e}")
            raise