import argparse
import datetime
import json
import nft_storage
import os
import requests
import time

from nft_storage.api import nft_storage_api

from blockfrost import BlockFrostApi, ApiUrls

from utils import upload_to_ipfs, IPFS_SCHEME

MAX_ATTEMPTS = 3
IPFS_SCHEME = 'ipfs://'
IPFS_GATEWAY = 'https://ipfs.io/ipfs'

NFT_STORAGE_API = 'https://api.nft.storage'

def get_metadata_images(metadata_dir):
    for filename in os.listdir(metadata_dir):
        with open(os.path.join(metadata_dir, filename), 'r') as file_obj:
            metadata_json = json.load(file_obj)
            for (nft_policy, nft_policy_objs) in metadata_json['721'].items():
                if nft_policy == 'version':
                    continue
                for nft_val in nft_policy_objs.values():
                    yield nft_val['image']

def get_assets_for(blockfrost_api, policy_id):
    page = 1
    while True:
        attempt = 1
        while True:
            try:
                assets = blockfrost_api.assets_policy(policy_id, page=page)
                break
            except Exception as e:
                if attempt >= MAX_ATTEMPTS:
                    raise e
                print(f"Retrying after exception: {e}")
                time.sleep(5)
                attempt += 1
        if not assets:
            return
        for asset in assets:
            metadata = blockfrost_api.asset(asset.asset).onchain_metadata
            if not metadata:
                print(f"{asset.asset} has no metadata, skipping...")
                continue
            yield metadata.image
        page += 1
        
def get_cid_for(image):
    normalized_str = ''.join(image)
    return normalized_str[len(IPFS_SCHEME):]

def migrate_to_nftstorage(asset_generator, output_dir, nft_storage_config, remove_temp_files, existing_uploads):
    for asset in asset_generator:
        cid = get_cid_for(asset)
        if cid in existing_uploads:
            print(f"Skipping {cid}...")
            continue
        output_car = f"{os.path.join(output_dir, cid)}.car"
        os.system(f"ipfs dag export {cid} > {output_car}")
        with open(output_car, 'wb') as output_file:
            print(f"[{datetime.datetime.now()}] Downloading {cid}...")
            output_file.write(requests.get(f"{IPFS_GATEWAY}/{cid}?format=car").content)
        with nft_storage.ApiClient(nft_storage_config, header_name='Content-Type', header_value='application/car') as api_client:
            upload_to_ipfs(api_client, output_car)
        if remove_temp_files:
            os.unlink(output_car)

def check_migration(asset_generator, nft_storage_config, existing_uploads):
    for asset in asset_generator:
        cid = get_cid_for(asset)
        if cid in existing_uploads:
            print(f"Skipping {cid}...")
            continue
        with nft_storage.ApiClient(nft_storage_config, header_name='Content-Type', header_value='application/car') as api_client:
            try:
                api_resp = requests.get(f"{NFT_STORAGE_API}/check/{cid}")
                api_resp_json = json.loads(api_resp.text)
                api_resp_found_msg = "has been uploaded" if api_resp_json['ok'] else "IS MISSING"
                print(f"{cid} {api_resp_found_msg}", flush=True)
            except nft_storage.ApiException as e:
                print("Exception when calling NFTStorageAPI->check: %s\n" % e)

def get_parser():
    parser = argparse.ArgumentParser(description='Migration Tools for NFT Images')
    subparsers = parser.add_subparsers(dest='command')

    nftstorage_parser = subparsers.add_parser('ipfs-to-nftstorage', help='Move NFTs from existing IPFS CARs to free storage on NFT.Storage (requires running IPFS daemon)')
    nftstorage_parser.add_argument('--output-dir', type=str, required=True)
    nftstorage_parser.add_argument('--blockfrost-key', type=str, required=True)
    nftstorage_parser.add_argument('--nft-storage-key', type=str, required=True)
    nftstorage_parser.add_argument('--remove-temp-files', action='store_true')
    nftstorage_parser.add_argument('--existing-attempt', type=str, required=False)

    nftstorage_input_group = nftstorage_parser.add_mutually_exclusive_group(required=True)
    nftstorage_input_group.add_argument('--policy', type=str)
    nftstorage_input_group.add_argument('--metadata-dir', type=str)

    check_parser = subparsers.add_parser('check-migration', help='Check that a policy or metadata set has been entirely migrated to NFT.Storage')
    check_parser.add_argument('--blockfrost-key', type=str, required=True)
    check_parser.add_argument('--nft-storage-key', type=str, required=True)
    check_parser.add_argument('--existing-attempt', type=str, required=False)

    check_input_group = check_parser.add_mutually_exclusive_group(required=True)
    check_input_group.add_argument('--policy', type=str)
    check_input_group.add_argument('--metadata-dir', type=str)

    return parser

if __name__ == '__main__':
    _args = get_parser().parse_args()
    blockfrost_api = BlockFrostApi(project_id=_args.blockfrost_key, base_url=ApiUrls.mainnet.value)
    asset_generator = []
    if _args.policy:
        asset_generator = get_assets_for(blockfrost_api, _args.policy)
    elif _args.metadata_dir:
        asset_generator = get_metadata_images(_args.metadata_dir)
    nft_storage_config = nft_storage.Configuration(access_token = _args.nft_storage_key)
    existing_uploads = open(_args.existing_attempt, 'r').read() if _args.existing_attempt else ''
    if _args.command == 'ipfs-to-nftstorage':
        migrate_to_nftstorage(asset_generator, _args.output_dir, nft_storage_config, _args.remove_temp_files, existing_uploads)
    elif _args.command == 'check-migration':
        check_migration(asset_generator, nft_storage_config, existing_uploads)