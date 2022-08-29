import binascii
import os
import requests
import subprocess

from cid import make_cid

TANGZ_POLICY = '33568ad11f93b3e79ae8dee5ad928ded72adcea719e92108caf1521b'
BLOCKFROST_KEY = os.environ['SET_YOUR_BLOCKFROST_PROJ_HERE']
EXCLUDED_DIRS = ['surgical mask redo/pics']

def get_from_blockfrost(asset_name):
    asset_name_hex = binascii.hexlify(asset_name.encode('ascii')).decode('ascii')
    #print(asset_name_hex)
    asset_data = requests.get(
        f"https://cardano-mainnet.blockfrost.io/api/v0/assets/{TANGZ_POLICY}{asset_name_hex}",
        headers={
            'project_id': BLOCKFROST_KEY,
            'Content-Type': 'application/json'
        }
    ).json()
    if 'status_code' in asset_data and asset_data['status_code'] == 404:
        return None
    return asset_data

def is_excluded(png_filename):
    for excluded_dir in EXCLUDED_DIRS:
        potentially_excluded_filename = os.path.join(excluded_dir, png_filename)
        if os.path.exists(potentially_excluded_filename):
            return potentially_excluded_filename
    return None

def get_cid_for(image_metadata):
    if image_metadata[0:7] == 'ipfs://':
        return make_cid(image_metadata[7:])
    return make_cid(image_metadata).to_v0()

def process_batch(prefix, start, finish):
    for item in range(start, finish):
        item_name = f"WildTangz {item}"
        png_filename = f"{item_name}.png"
        print(f"Computing information for '{item_name}'...", flush=True)
        potentially_excluded_filename = is_excluded(png_filename)
        pic = os.path.join(prefix, png_filename)
        if potentially_excluded_filename:
            print(f"Updating to substitute at '{potentially_excluded_filename}'")
            pic = potentially_excluded_filename
        if not os.path.exists(pic):
            print(f"Could not find Wild Tangz picture '{pic}'")
        stdout = subprocess.check_output(['ipfs', 'add', '-n', '--chunker=size-262144', '--cid-version=1', pic], stderr=subprocess.DEVNULL)
        v1_cid_str = stdout.split()[1]
        v0_cid = make_cid(v1_cid_str.decode('utf-8').strip()).to_v0()
        minted_pic = get_from_blockfrost(item_name)
        if not minted_pic:
            print(f"'{item_name}' was never minted, continuing...")
            continue
        tangz_cid = get_cid_for(minted_pic['onchain_metadata']['image'])
        if v0_cid != tangz_cid:
            print(f"Invalid CIDs for '{item_name}':\n\tComputed {v0_cid}\n\tFound {tangz_cid}")

process_batch('batch 1/pics', 1, 1724)
process_batch('batch 2/pics', 1722, 3371)
process_batch('batch 3/pics', 3370, 3700)
process_batch('batch 4/pics', 3699, 4410)
