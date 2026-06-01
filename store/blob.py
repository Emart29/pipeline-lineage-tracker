import io
import asyncio
from minio import Minio
from minio.error import S3Error
from config import settings


class BlobStore:
    def __init__(self):
        self._client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False,
        )
        self._bucket = settings.MINIO_BUCKET
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def upload_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self._client.put_object(
            self._bucket,
            key,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return key

    def download_bytes(self, key: str) -> bytes:
        response = self._client.get_object(self._bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def object_exists(self, key: str) -> bool:
        try:
            self._client.stat_object(self._bucket, key)
            return True
        except S3Error:
            return False

    def delete_object(self, key: str) -> None:
        self._client.remove_object(self._bucket, key)

    def list_objects(self, prefix: str = "") -> list[str]:
        objects = self._client.list_objects(self._bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects]

    async def async_upload_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        return await asyncio.to_thread(self.upload_bytes, key, data, content_type)

    async def async_download_bytes(self, key: str) -> bytes:
        return await asyncio.to_thread(self.download_bytes, key)

    async def async_object_exists(self, key: str) -> bool:
        return await asyncio.to_thread(self.object_exists, key)
