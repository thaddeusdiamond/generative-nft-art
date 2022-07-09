import argparse
import json
import os
import pprint
import random
import shutil
import time

import nft_storage

from cid import make_cid
from nft_storage.api import nft_storage_api
from PIL import Image

# Image Generation Constants

# https://dribbble.com/shots/843152-Colors?list=searches&tag=color_palette
# Names from https://colors.artyclick.com/color-name-finder/
BG_COLORS = {
        'Pure White': (255, 255, 255),
        'Deep Gray': (126, 125, 129),
        'Tobacco Brown': (119, 91, 63),
        'Blue Koi': (92, 163, 212),
        'Fountain Blue': (97, 188, 183),
        'Avocado Green': (181, 193, 71),
        'Yellowish Orange': (237, 165, 70),
        'Halloween Orange': (229, 107, 54),
        'Orangy Red': (202, 70, 64),
        'Carmine Pink': (222, 109, 124),
        'Amethyst Purple': (154, 104, 210)
}
RARITY_MAPPING = {
        'limited edition': 0,
        'extremely rare': 1,
        'very rare': 5,
        'rare': 10,
        'common': 20
}
EXCLUSIONARY_TRAITS = {
    'Angry.png': ['Gameday Paint.png', 'Skeleton Facepaint.png'],
    'Astronaut.png': ['Snorkel.png', 'Angel Wings.png', 'VR Headset.png'],
    'Cyclops.png': ['Monocle.png', 'Aviators.png', 'Glasses.png', 'Laserbeam.png', 'Snorkel.png', 'Sunglasses.png', 'Swim Goggles.png'],
    'Headphones.png': ['Earpods.png'],
    'Ski Goggles.png': ['Casino Visor.png', 'Headlamp.png'],
    'Snorkel.png': ['Beanie.png', 'Clover Ballcap.png', 'Headphones.png', 'MATA Ballcap.png', 'Purple and Gold Ballcap.png', 'Rasta.png', 'Yellow Ballcap.png'],
    'VR Headset.png': ['Beanie.png', 'Casino Visor.png', 'Clover Ballcap.png', 'Earpods.png', 'Headband.png', 'Headlamp.png', 'Headphones.png', 'MATA Ballcap.png', 'Purple and Gold Ballcap.png', 'Yellow Ballcap.png'],
    'Zombie.png': ['Baseball Uniform.png', 'Bomber Jacket.png', 'Business Suit.png', 'Flannel.png', 'Hawaiian Shirt.png', 'Judge Robes.png', 'Pajamas.png', 'Turtleneck Sweater.png', 'Crewneck Tee.png', 'Workout Tracksuit.png']
}

# Image creation constants
OUTPUT_DIR = 'output'
OUTPUT_SIZE = (2100, 2100)
ORDERED_SUBPICS_DIRS = ['fur', 'body', 'accessories', 'eyes', 'eyewear', 'headwear', 'clothing', 'mouth']
PICS_DIR = 'pics'
REQUIRED_SUBPICS = ['fur', 'body', 'mouth', 'eyes']

# IPFS-related constants
LOCALHOST = '127.0.0.1'
IPFS_PORT = 5001

# NFT Storage Constants
TANGZ_POLICY_ID = '33568ad11f93b3e79ae8dee5ad928ded72adcea719e92108caf1521b'
NFTSTORAGE_CONFIG = nft_storage.Configuration(access_token = os.environ.get('NFTSTORAGE_KEY'))

def upload_to_ipfs_new_metadata(in_pics_dir, in_metadata_dir, out_metadata_dir):
    metadata_list = load_metadata(in_metadata_dir)
    nfts_list = [unwrap(nft) for nft in metadata_list]
    with nft_storage.ApiClient(NFTSTORAGE_CONFIG) as api_client:
        print("Beginning NFT.Storage file uploads")
        api_instance = nft_storage_api.NFTStorageAPI(api_client)
        for nft in nfts_list:
            nft['image'] = upload_to_ipfs(api_client, nft, in_pics_dir)
            dump_metadata_files([nft], out_metadata_dir)

def generate_sample_images(metadata_dir, pics_dir, num_images, num_iterations, output_pics_dir):
    metadata_list = load_metadata(metadata_dir)
    random_set = random.choices(metadata_list, k=num_images)
    for iteration in range(num_iterations):
        print(f"Beginning optimization iteration {iteration + 1}...")
        for index in range(len(random_set)):
            hamming_distances = [unwrap_calc_hamming(random_set[index], nft) for nft in random_set if nft != random_set[index]]
            current_min_hamming = min(hamming_distances)
            for nft in metadata_list:
                hamming_distances = [unwrap_calc_hamming(random_nft, nft) for random_nft in random_set if random_nft != random_set[index]]
                new_min_hamming = min(hamming_distances)
                if new_min_hamming > current_min_hamming:
                    print(f"Replacing at index {index} ({current_min_hamming} -> {new_min_hamming})")
                    random_set[index] = nft
                    break
    min_hamming = 1000
    for i in range(num_images):
        for j in range(i + 1, num_images):
            min_hamming = min(min_hamming, unwrap_calc_hamming(random_set[i], random_set[j]))
    random_set_ids = sorted([unwrap(nft)['name'] for nft in random_set])
    print(f"Recommending the following for samples (min hamming: {min_hamming}) {random_set_ids}")
    for random_id in random_set_ids:
        png_filename = get_pic_filename(random_id)
        try:
            shutil.copy(os.path.join(pics_dir, png_filename), os.path.join(output_pics_dir, png_filename))
        except FileNotFoundError as e:
            print(e)

def compose_image(combination, bg_color, in_pics_dir):
    background = Image.new('RGBA', OUTPUT_SIZE, bg_color)
    for subpic_dir in ORDERED_SUBPICS_DIRS:
        if combination[subpic_dir]:
            filename = os.path.join(in_pics_dir, subpic_dir, combination[subpic_dir])
            foreground = Image.open(filename).convert('RGBA')
            background = Image.alpha_composite(background, foreground)
    return background

def generate_image(nft, in_pics_dir, out_pics_dir):
    bg_color = BG_COLORS[nft['background']]
    try:
        composed_image = compose_image(nft, bg_color, in_pics_dir)
        composed_image.save(os.path.join(out_pics_dir, get_pic_filename(nft['name'])))
    except FileNotFoundError as e:
        print(e)

def calculate_hamming_distance(nft_a, nft_b):
    hamming_distance = 0
    for subpic_dir in ORDERED_SUBPICS_DIRS:
        if nft_a[subpic_dir] != nft_b[subpic_dir]:
            hamming_distance += 1
    return hamming_distance

def get_nft_name(i):
    return f"WildTangz {i+1}"

def get_pic_filename(nft_name):
    return f"{nft_name}.png"

def get_metadata_filename(nft_name):
    return f"{nft_name}.json"

def wrap(nft_name, values):
    return {
        "721": {
            TANGZ_POLICY_ID: {
                nft_name: values
            },
            "version": "1.0"
        }
    }

def unwrap(metadata):
    return next(iter(metadata["721"][TANGZ_POLICY_ID].values()))

def get_metadata_for(nft):
    metadata = { "mediaType": "image/png", "project": "Wild Tangz" }
    for key in nft:
        metadata[key] = os.path.splitext(nft[key])[0]
    return wrap(nft['name'], metadata)

def write_metadata_statistics(full_list, metadata_statistics, expected_ratios, statistics_file):
    total_nfts = len(full_list)
    total_hamming_distance = 0
    min_hamming_distance = len(ORDERED_SUBPICS_DIRS)
    max_hamming_distance = 0
    for i in range(0, total_nfts):
        for j in range(i + 1, total_nfts):
            hamming_distance = calculate_hamming_distance(full_list[i], full_list[j])
            min_hamming_distance = min(min_hamming_distance, hamming_distance)
            max_hamming_distance = max(max_hamming_distance, hamming_distance)
            total_hamming_distance += hamming_distance
    avg_hamming_distance = total_hamming_distance / (total_nfts * (total_nfts + 1) / 2.0)
    statistics_file.write(f"Minimum hamming distance: {min_hamming_distance}\n")
    statistics_file.write(f"Maximum hamming distance: {max_hamming_distance}\n")
    statistics_file.write(f"Average hamming distance: {avg_hamming_distance}\n")
    statistics_file.write(f"\tNAME                               ACTUAL      EXPECTED   VARIANCE\n")
    for subpic_dir in metadata_statistics.keys():
        statistics_file.write(f"{subpic_dir} -----------------\n")
        for item in metadata_statistics[subpic_dir].keys():
            total_number = metadata_statistics[subpic_dir][item]
            filename = os.path.basename(item) if item else 'None'
            ratio = total_number / float(total_nfts) * 100.0
            expected = expected_ratios[subpic_dir][filename] * 100.0 if item else 0.0
            variance = int(((ratio / expected) - 1) * 100) if expected else 1000.0
            statistics_file.write(f"\t{filename:30}{ratio:-10.1f}%{expected:-10.1f}%{variance:8}%\n")

def dump_metadata_files(full_list, metadata_dir):
    for nft in full_list:
        with open(os.path.join(metadata_dir, get_metadata_filename(nft['name'])), 'w') as metadata_file:
            json.dump(get_metadata_for(nft), metadata_file)

def compile_metadata_statistics(full_list, variable_traits):
    metadata_statistics = {}
    for nft in full_list:
        for subpic_dir in variable_traits:
            if not subpic_dir in metadata_statistics:
                metadata_statistics[subpic_dir] = {}
            nft_subpic = nft[subpic_dir]
            if not nft_subpic in metadata_statistics[subpic_dir]:
                metadata_statistics[subpic_dir][nft_subpic] = 0
            metadata_statistics[subpic_dir][nft_subpic] += 1
    return metadata_statistics

def is_illegal(nft_combination):
    traits = nft_combination.values()
    for excluder in EXCLUSIONARY_TRAITS:
        for exclusion in EXCLUSIONARY_TRAITS[excluder]:
            if excluder in traits and exclusion in traits:
                #print(f"Excluding combination with '{excluder}' due to '{exclusion}'")
                return True
    return False


def generate_nft_metadata(start_num, total_nfts, traits_ratios, acceptable_hamming, static_traits, variable_traits):
    full_list = []
    last_printed = time.time()
    for i in range(start_num, start_num + total_nfts):
        nft_combination = {}
        while not nft_combination:
            if time.time() - last_printed > 5:
                print(f"Found {i} combinations so far...")
                last_printed = time.time()
            for key in static_traits:
                nft_combination[key] = static_traits[key]
            for subpic_dir in variable_traits:
                nft_combination[subpic_dir] = None
                dice_roll = random.random()
                for filename, ratio in traits_ratios[subpic_dir].items():
                    if dice_roll < ratio:
                        nft_combination[subpic_dir] = filename
                        break
                    dice_roll -= ratio
                if not nft_combination[subpic_dir] and subpic_dir in REQUIRED_SUBPICS:
                    print(f"Illegal state: required subpic {subpic_dir} not found (roll {dice_roll})")
            if is_illegal(nft_combination):
                nft_combination = {}
                continue
            for already_found in full_list:
                if calculate_hamming_distance(nft_combination, already_found) < acceptable_hamming:
                    #print(f"Repeating iteration {i}...")
                    nft_combination = {}
                    break
        nft_combination['name'] = get_nft_name(i)
        full_list.append(nft_combination)
    return full_list

def add_backgrounds_to(nft_metadata):
    for metadata in nft_metadata:
        metadata['background'] = random.choice(list(BG_COLORS.keys()))

def generate_nfts(start_num, total_nfts, traits_ratios, in_pics_dir, acceptable_hamming, output_pics_dir, metadata_dir, static_traits, variable_traits):
    nft_metadata = generate_nft_metadata(start_num, total_nfts, traits_ratios, acceptable_hamming, static_traits, variable_traits)
    add_backgrounds_to(nft_metadata)
    dump_metadata_files(nft_metadata, metadata_dir) # Temporary in case something happens during the composition run
    metadata_statistics = compile_metadata_statistics(nft_metadata, variable_traits)
    with open(os.path.join(os.path.dirname(metadata_dir), 'metadata_statistics.tsv'), 'w') as statistics_file:
        write_metadata_statistics(nft_metadata, metadata_statistics, traits_ratios, statistics_file)
    with nft_storage.ApiClient(NFTSTORAGE_CONFIG) as api_client:
        for nft in nft_metadata:
            generate_image(nft, in_pics_dir, output_pics_dir)
            nft['image'] = upload_to_ipfs(api_client, nft, output_pics_dir)
            dump_metadata_files([nft], metadata_dir)

def unwrap_calc_hamming(metadata_a, metadata_b):
    return calculate_hamming_distance(unwrap(metadata_a), unwrap(metadata_b))

def load_metadata(metadata_dir):
    metadata_list = []
    for filename in sorted(os.listdir(metadata_dir)):
        metadata_filename = os.path.join(metadata_dir, filename)
        with open(metadata_filename, 'r') as metadata_file:
            nft_metadata = json.load(metadata_file)
            metadata_list.append(nft_metadata)
            if not nft_metadata:
                print(metadata_filename)
    return metadata_list

def upload_to_ipfs(api_client, nft, in_pics_dir):
    api_instance = nft_storage_api.NFTStorageAPI(api_client)
    nft_filename = os.path.join(in_pics_dir, get_pic_filename(nft['name']))
    print(f"Uploading {nft_filename}")
    tries_remaining = 3
    while tries_remaining:
        with open(nft_filename, 'rb') as nft_body:
            try:
                # https://github.com/nftstorage/python-client/issues/1
                tries_remaining -= 1
                api_response = api_instance.store(nft_body, _check_return_type=False)
                #print(f"Received response: {api_response}")
                print(f"Uploaded as {api_response.value['cid']}")
                return f"ipfs://{make_cid(api_response.value['cid']).to_v0()}"
                #print(f"Dumping {nft} to file")
            except nft_storage.ApiException as e:
                print("Exception when calling NFTStorageAPI->store: %s\n...Retrying" % e)
                time.sleep(1)
    raise ValueError('Could not upload to IPFS, see exception log.')

def check_uniqueness(nft_metadata, reference_dir):
    with open(nft_metadata, 'r') as nft_file:
        nft = json.load(nft_file)
    print(nft)
    reference_set = load_metadata(reference_dir)
    hamming_distances = [(unwrap_calc_hamming(nft, reference_nft), reference_nft) for reference_nft in reference_set]
    min_hamming = 10000
    for distance, ref_nft in hamming_distances:
        if distance < min_hamming:
            min_hamming = distance
            reference = ref_nft
    print(f"Min hamming found {min_hamming}:\n{reference}")
    #print(unwrap_calc_hamming(nft, hamming_distances[i][1]))

def regenerate_image(nft_metadata_filename, in_pics_dir, output_pics_dir):
    with open(nft_metadata_filename, 'r') as nft_metadata_file:
        nft = unwrap(json.load(nft_metadata_file))
        for key in nft:
            if key in ORDERED_SUBPICS_DIRS and not nft[key].lower().endswith('.png'):
                nft[key] = f"{nft[key]}.png"
        generate_image(nft, in_pics_dir, output_pics_dir)

def read_percentages(percentages_filename, desired_categories):
    percentages = {}
    with open(_args.percentages_file, 'r') as percentages_file:
        traits_descriptions = json.load(percentages_file)
        for category in traits_descriptions:
            category_ratios = {}
            if not category in desired_categories:
                raise ValueError(f"Unexpected category '{category}' found in percentages file")
            category_traits = traits_descriptions[category]
            mapped_freqs = [RARITY_MAPPING[category_traits[attr]] for attr in category_traits]
            total_freqs = float(sum(mapped_freqs))
            percentages[category] = dict([(attr, RARITY_MAPPING[category_traits[attr]] / total_freqs) for attr in category_traits])
    return percentages

def make_and_get_output_dir(prefix):
    output_dir = os.path.join(OUTPUT_DIR, str(int(time.time())))
    metadata_dir = os.path.join(output_dir, 'metadata')
    pics_dir = os.path.join(output_dir, 'pics')
    [os.makedirs(out_dir) for out_dir in [output_dir, metadata_dir, pics_dir]]
    return (pics_dir, metadata_dir)

def get_parser():
    parser = argparse.ArgumentParser(description='Generate NFTs for Wild Tangz')
    subparsers = parser.add_subparsers(dest='command')

    nfts_parser = subparsers.add_parser('generate-nfts', help='Compose NFTs according to trait gradient algorithm')
    nfts_parser.add_argument('--total-nfts', type=int, required=True)
    nfts_parser.add_argument('--percentages-file', required=True)
    nfts_parser.add_argument('--min-hamming', type=int, required=True)
    nfts_parser.add_argument("--static-trait", action='append', type=lambda kv: kv.split("="), dest='static_traits')
    nfts_parser.add_argument('--start-num', type=int, required=False, default=0)

    samples_parser = subparsers.add_parser('generate-samples', help='Generate recommendations for unique samples')
    samples_parser.add_argument('--metadata-dir', required=True)
    samples_parser.add_argument('--pics-dir', required=True)
    samples_parser.add_argument('--num-images', type=int, required=True)
    samples_parser.add_argument('--num-iterations', type=int, required=True)

    ipfs_parser = subparsers.add_parser('upload-to-ipfs', help='Upload raw images to IPFS')
    ipfs_parser.add_argument('--metadata-dir', required=True)
    ipfs_parser.add_argument('--pics-dir', required=True)

    ipfs_parser = subparsers.add_parser('check-uniqueness', help='Compare the uniqueness of this NFT to the reference set')
    ipfs_parser.add_argument('--nft-metadata', required=True)
    ipfs_parser.add_argument('--reference-set-dir', required=True)

    ipfs_parser = subparsers.add_parser('regenerate-image', help='Regenerate an image based on metadata file provided (useful for 1-of-1s)')
    ipfs_parser.add_argument('--nft-metadata-file', required=True)

    return parser

def seed_random():
    random.seed(12345)

if __name__ == '__main__':
    seed_random()
    _args = get_parser().parse_args()
    if not _args.command in ['check-uniqueness']:
        _pics_dir, _metadata_dir = make_and_get_output_dir(OUTPUT_DIR)
    if _args.command == 'generate-nfts':
        _traits_ratios = read_percentages(_args.percentages_file, ORDERED_SUBPICS_DIRS)
        _static_traits = dict(_args.static_traits) if _args.static_traits else {}
        _variable_traits = [trait for trait in ORDERED_SUBPICS_DIRS if not (trait in _static_traits.keys())]
        print(f"Creating {_args.total_nfts} NFT w/hamming >{_args.min_hamming} & ratios from '{_args.percentages_file}' to dir '{_pics_dir}'")
        print(f"Static traits specified: {_static_traits}")
        print(f"Variable traits specified: {_variable_traits}")
        generate_nfts(_args.start_num, _args.total_nfts, _traits_ratios, PICS_DIR, _args.min_hamming, _pics_dir, _metadata_dir, _static_traits, _variable_traits)
    elif _args.command == 'generate-samples':
        generate_sample_images(_args.metadata_dir, _args.pics_dir, _args.num_images, _args.num_iterations, _pics_dir)
    elif _args.command == 'upload-to-ipfs':
        upload_to_ipfs_new_metadata(_args.pics_dir, _args.metadata_dir, _metadata_dir)
    elif _args.command == 'check-uniqueness':
        check_uniqueness(_args.nft_metadata, _args.reference_set_dir)
    elif _args.command == 'regenerate-image':
        regenerate_image(_args.nft_metadata_file, PICS_DIR, _pics_dir)
    else:
        raise ValueError("No command passed to the program.  Use -h for help.")
