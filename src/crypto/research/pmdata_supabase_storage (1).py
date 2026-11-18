#!/usr/bin/env python3
"""Supabase Storage archive for PMData Polymarket research (server-side only).

Requirements: boto3, requests. Supabase S3 protocol must be enabled and a
PRIVATE Storage bucket must already exist. Source archives stream PMData ->
Codespace RAM buffers -> Supabase Storage multipart S3; never write ZIPs to
local disk. Archived ZIP/Parquet objects are read via S3 HTTP byte ranges so
backtests only fetch the specific Parquet members they need.

Codespace secrets: SUPABASE_S3_ENDPOINT, SUPABASE_S3_REGION,
SUPABASE_S3_ACCESS_KEY_ID, SUPABASE_S3_SECRET_ACCESS_KEY.
Optional: PMDATA_SUPABASE_BUCKET (default prediction-market-data).
Do not expose these credentials to browsers or commit them to Git.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

# Load project-root .env for Windows/Linux when running scripts directly.
# An already configured process environment takes priority.
try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise RuntimeError("Install .env loader: python -m pip install python-dotenv") from exc
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
import re
import zipfile

MB = 1024 * 1024


class S3RangeReader(io.RawIOBase):
    """Seekable S3 object reader with small in-memory block cache."""
    def __init__(self, s3, bucket, key, *, min_fetch=2*MB):
        self.s3 = s3
        self.bucket = bucket
        self.key = key
        meta = s3.head_object(Bucket=bucket, Key=key)
        self.length = int(meta['ContentLength'])
        self.etag = meta.get('ETag')
        self.pos = 0
        self.start = -1
        self.cache = b''
        self.min_fetch = min_fetch
        super().__init__()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        self._checkClosed()
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self._checkClosed()
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self.pos + offset
        elif whence == io.SEEK_END:
            new = self.length + offset
        else:
            raise ValueError('Unknown seek mode')
        if new < 0:
            raise ValueError('Negative seek')
        self.pos = int(new)
        return self.pos

    def read(self, n=-1):
        self._checkClosed()
        if self.pos >= self.length:
            return b''
        if n is None or n < 0:
            n = self.length-self.pos
        n = min(int(n), self.length-self.pos)
        if n <= 0:
            return b''
        if self.start <= self.pos and self.pos+n <= self.start+len(self.cache):
            data = self.cache[self.pos-self.start:self.pos-self.start+n]
        else:
            start = self.pos
            # Download no more than the requested size or a small look-ahead.
            end = min(self.length, start + max(n, self.min_fetch)) - 1
            args=dict(Bucket=self.bucket, Key=self.key, Range=f'bytes={start}-{end}')
            if self.etag:
                args['IfMatch'] = self.etag
            response = self.s3.get_object(**args)
            try:
                data_bytes = response['Body'].read()
            finally:
                response['Body'].close()
            expected = end-start+1
            if len(data_bytes) != expected:
                raise IOError(f'S3 returned {len(data_bytes)} of {expected} bytes for {self.key}')
            self.start = start
            self.cache = data_bytes
            data = data_bytes[:n]
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


class SupabaseArchive:
    def __init__(self, *, bucket=None):
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError('Install Supabase Storage dependency: python3 -m pip install boto3') from exc
        endpoint = os.getenv('SUPABASE_S3_ENDPOINT', '').rstrip('/')
        region = os.getenv('SUPABASE_S3_REGION', '')
        access = os.getenv('SUPABASE_S3_ACCESS_KEY_ID', '')
        secret = os.getenv('SUPABASE_S3_SECRET_ACCESS_KEY', '')
        if not all((endpoint,region,access,secret)):
            raise RuntimeError('Supabase Storage S3 settings missing: add SUPABASE_S3_ENDPOINT, '
                               'SUPABASE_S3_REGION, SUPABASE_S3_ACCESS_KEY_ID, '
                               'SUPABASE_S3_SECRET_ACCESS_KEY to your project .env or process environment. '
                               'PREDICTION_DATABASE_URL is for Postgres and cannot authenticate Storage.')
        if not endpoint.startswith('https://') or not endpoint.endswith('/storage/v1/s3'):
            raise RuntimeError('SUPABASE_S3_ENDPOINT must be https://<project>.storage.supabase.co/storage/v1/s3 '
                               '(or the S3 endpoint shown in Supabase Storage settings)')
        self.bucket = bucket or os.getenv('PMDATA_SUPABASE_BUCKET', 'prediction-market-data')
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint, region_name=region,
            aws_access_key_id=access, aws_secret_access_key=secret,
            config=Config(signature_version='s3v4',
                          retries={'max_attempts':5,'mode':'standard'},
                          connect_timeout=20,read_timeout=180,
                          s3={'addressing_style':'path'}))
        # Fail early rather than unexpectedly writing back to local disk.
        self.s3.head_bucket(Bucket=self.bucket)

    @staticmethod
    def object_key(dataset, filename):
        """Stable deterministic paths independent of backtest price parameters."""
        if dataset not in ('btc-1h','btc-5m'):
            raise ValueError('Unknown PMData dataset')
        name = Path(filename).name
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise ValueError('Unsafe object filename')
        return f'pmdata/polymarket/{dataset}/{name}'

    def exists(self, key):
        from botocore.exceptions import ClientError
        try:
            return self.s3.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            status = exc.response.get('ResponseMetadata',{}).get('HTTPStatusCode')
            if status == 404:
                return None
            raise

    def transfer_pmdata(self, session, pmdata_key, url, dataset, filename, *, offline=False, allow_404=False):
        """Send PMData response directly to S3 multipart upload; RAM only."""
        key = self.object_key(dataset,filename)
        meta = self.exists(key)
        if meta is not None:
            if meta['ContentLength'] == 0:
                raise RuntimeError(f'Existing empty S3 object: {key}; investigate before reusing')
            print(f'SUPABASE REUSE {key}',flush=True)
            return key
        if offline:
            raise RuntimeError(f'Object missing from Supabase Storage (--offline): {key}')
        if not pmdata_key:
            raise RuntimeError('PMDATA_API_KEY required to fetch an uncached object')
        try:
            from boto3.s3.transfer import TransferConfig
            from botocore.exceptions import ClientError
        except ImportError as exc:
            raise RuntimeError('boto3 installation incomplete') from exc
        print(f'STREAM PMData -> Supabase Storage: {key}',flush=True)
        with session.get(url, headers={'api_key': pmdata_key},stream=True,
                         timeout=(20,240)) as response:
            if response.status_code == 404 and allow_404:
                return None
            if response.status_code != 200:
                raise RuntimeError(f'PMData HTTP {response.status_code}: {response.text[:260]}')
            content_length=response.headers.get('Content-Length')
            # Network streaming, not a local temporary ZIP. Disable transparent
            # content decoding; PMData sends the archive's actual bytes.
            response.raw.decode_content = False
            self.s3.upload_fileobj(
                response.raw, self.bucket, key,
                ExtraArgs={'ContentType':'application/zip' if filename.endswith('.zip')
                           else 'application/octet-stream'},
                Config=TransferConfig(multipart_threshold=6*MB,
                                      multipart_chunksize=8*MB,
                                      max_concurrency=1,use_threads=False))
        meta = self.exists(key)
        if meta is None:
            raise RuntimeError(f'Upload not visible in Supabase Storage: {key}')
        if content_length and meta['ContentLength'] != int(content_length):
            # Do NOT retain truncated historical archives.
            self.s3.delete_object(Bucket=self.bucket, Key=key)
            raise IOError(f'Content-Length mismatch after upload for {key}; removed corrupt object')
        self.verify_magic(key, b'PK\x03\x04' if filename.endswith('.zip') else b'PAR1')
        print(f'  STORED {meta["ContentLength"]:,} bytes (no local archive)',flush=True)
        return key

    def verify_magic(self,key,magic):
        obj=self.s3.get_object(Bucket=self.bucket,Key=key,Range='bytes=0-3')
        try:
            actual=obj['Body'].read()
        finally:
            obj['Body'].close()
        if actual != magic:
            raise IOError(f'Unexpected file header in Supabase Storage object {key}: {actual!r}')

    def transfer_local_cache(self, dataset, path):
        """One-time migration of existing Codespaces archives, without PMData downloads."""
        from boto3.s3.transfer import TransferConfig
        path=Path(path)
        key=self.object_key(dataset,path.name)
        if self.exists(key) is not None:
            print(f'SUPABASE REUSE {key}',flush=True)
            return key
        if not path.is_file() or path.stat().st_size==0:
            raise FileNotFoundError(path)
        self.s3.upload_file(str(path),self.bucket,key,
                            ExtraArgs={'ContentType':'application/zip' if path.suffix=='.zip'
                                       else 'application/octet-stream'},
                            Config=TransferConfig(multipart_threshold=6*MB,
                                                  multipart_chunksize=8*MB,
                                                  max_concurrency=2))
        remote=self.exists(key)
        if remote is None or remote['ContentLength'] != path.stat().st_size:
            raise IOError(f'Migration upload mismatch: {key}')
        self.verify_magic(key,b'PK\x03\x04' if path.suffix=='.zip' else b'PAR1')
        print(f'MIGRATED {path} -> {key}',flush=True)
        return key

    def remote_reader(self, key):
        return S3RangeReader(self.s3,self.bucket,key)

    def parquet(self,key):
        import pandas as pd
        # pyarrow uses seek/read; S3 byte ranges, never a persistent local file.
        stream = self.remote_reader(key)
        try:
            return pd.read_parquet(stream)
        finally:
            stream.close()

    def open_zip(self,key):
        r=self.remote_reader(key)
        try:
            return zipfile.ZipFile(r,'r')
        except Exception:
            r.close()
            raise

    def put_result(self,path, key):
        """Tiny CSV/JSON research artifacts copied to cloud for reproducibility."""
        content='application/json' if str(path).endswith('.json') else 'text/csv'
        self.s3.upload_file(str(path),self.bucket,key,ExtraArgs={'ContentType':content})
        print(f'SAVED RESULT supabase://{self.bucket}/{key}',flush=True)


def migrate_cache(archive, dataset, root):
    root=Path(root)
    candidates=sorted(root.glob('*.zip'))+sorted(root.glob('*.parquet'))
    if not candidates:
        print(f'No existing local ZIP/Parquet objects to migrate in {root}')
    for p in candidates:
        archive.transfer_local_cache(dataset,p)
    print(f'Migration complete: {len(candidates)} files. Verify in Supabase dashboard '
          'before deleting the old local cache. Nothing deleted automatically.')


def self_test():
    """Unit tests for RangeReader and path naming, without credentials or networking."""
    class Body:
        def __init__(self,data): self.data=data
        def read(self): return self.data
        def close(self): pass
    class FakeS3:
        data=b'0123456789'*100
        def head_object(self,**kw): return dict(ContentLength=len(self.data),ETag='"fake"')
        def get_object(self,**kw):
            m=re.fullmatch(r'bytes=(\d+)-(\d+)',kw['Range'])
            a,b=map(int,m.groups())
            return {'Body':Body(self.data[a:b+1])}
    a=S3RangeReader(FakeS3(),'bucket','x',min_fetch=8)
    assert a.read(4)==b'0123'
    assert a.read(4)==b'4567'
    assert a.seek(-3,io.SEEK_END)==997
    assert a.read()==b'789'
    assert a.seek(50)==50 and a.read(7)==b'0123456'
    assert SupabaseArchive.object_key('btc-5m','2026-10-12_l2_Yes_btc-5m.zip')=='pmdata/polymarket/btc-5m/2026-10-12_l2_Yes_btc-5m.zip'
    try:
        SupabaseArchive.object_key('btc-5m','bad name.zip')
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid name accepted')
    print('PASS: Supabase storage range reads, offset seeks, cloud object naming; no network')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Supabase Storage PMData archival helper')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--migrate-cache',type=Path,
                        help='One-time cloud migration of preexisting local cache directory')
    parser.add_argument('--dataset',choices=('btc-1h','btc-5m'))
    opts=parser.parse_args()
    if opts.migrate_cache:
        if not opts.dataset:
            parser.error('--dataset is required with --migrate-cache')
        migrate_cache(SupabaseArchive(),opts.dataset,opts.migrate_cache)
    else:
        self_test()
