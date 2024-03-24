import datetime
import nft_storage
import time

from nft_storage.api import nft_storage_api

IPFS_SCHEME = 'ipfs://'

def upload_to_ipfs(api_client, filename):
    api_instance = nft_storage_api.NFTStorageAPI(api_client)
    print(f"[{datetime.datetime.now()}] Uploading {filename}...")
    tries_remaining = 3
    try:
        while tries_remaining:
            with open(filename, 'rb') as nft_body:
                try:
                    # https://github.com/nftstorage/python-client/issues/1
                    tries_remaining -= 1
                    api_response = api_instance.store(nft_body, _check_return_type=False)
                    #print(f"Received response: {api_response}")
                    print(f"[{datetime.datetime.now()}] Uploaded as {api_response.value['cid']}", flush=True)
                    return api_response.value['cid']
                    #print(f"Dumping {nft} to file")
                except nft_storage.ApiException as e:
                    print("Exception when calling NFTStorageAPI->store: %s\n...Retrying" % e)
                    time.sleep(1)
    except Exception as e:
        print(f"Encountered fatal exception for '{filename}': {e}")