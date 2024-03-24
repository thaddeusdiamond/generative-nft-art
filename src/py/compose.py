import argparse
import base64
import copy
import json
import numbers
import os
import pprint
import random
import re
import shutil
import time

import nft_storage

from cid import make_cid

from PIL import Image

from utils import IPFS_SCHEME, upload_to_ipfs

# Image Generation Constants
RARITY_MAPPING = {
    'limited edition': 0,
    'extremely rare': 1,
    'very rare': 5,
    'rare': 10,
    'common': 25,
    'supermajority': 66
}
FILE_FORMAT = 'png'

def generate_new_metadata_on_chain(in_metadata_dir, in_pics_dir, out_metadata_dir, policy, project):
    metadata_list = load_metadata(in_metadata_dir)
    nfts_list = [unwrap(nft, policy) for nft in metadata_list]
    onchain_nfts = []
    for nft in nfts_list:
        onchain_nft = copy.deepcopy(nft)
        nft_filename = os.path.join(in_pics_dir, get_pic_filename(nft['name']))
        with open(nft_filename, 'rb') as nft_file:
            base64_str = base64.b64encode(nft_file.read()).decode('ascii')
            base64_attr = f"data:image/png;base64,{base64_str}"
            onchain_nft['image'] = [base64_attr[i:i+64] for i in range(0, len(base64_attr), 64)]
        onchain_nfts.append(onchain_nft)
    dump_metadata_files(onchain_nfts, out_metadata_dir, policy, project)

def upload_to_ipfs_new_metadata(in_pics_dir, in_metadata_dir, out_metadata_dir, policy, project, nftstorage_key, existing_uploads_dir, cip_25):
    metadata_list = load_metadata(in_metadata_dir)
    existing_uploads = [] if not existing_uploads_dir else os.listdir(existing_uploads_dir)
    nfts_list = [unwrap(nft, policy) for nft in metadata_list]
    nftstorage_config = nft_storage.Configuration(access_token = nftstorage_key)
    with nft_storage.ApiClient(nftstorage_config) as api_client:
        print("Beginning NFT.Storage file uploads")
        for nft in nfts_list:
            if get_metadata_filename(nft['name']) in existing_uploads:
                continue
            image_ipfs = upload_to_ipfs(api_client, os.path.join(in_pics_dir, get_pic_filename(nft['name'])))
            if image_ipfs:
                nft['image'] = [IPFS_SCHEME, image_ipfs] if cip_25 else f"{IPFS_SCHEME}{image_ipfs}"
                dump_metadata_files([nft], out_metadata_dir, policy, project)

def generate_sample_images(metadata_dir, pics_dir, num_images, num_iterations, output_pics_dir, ordered_subpics_dirs, policy):
    metadata_list = load_metadata(metadata_dir)
    random_set = random.choices(metadata_list, k=num_images)
    for iteration in range(num_iterations):
        print(f"Beginning optimization iteration {iteration + 1}...")
        for index in range(len(random_set)):
            hamming_distances = [unwrap_calc_hamming(random_set[index], nft, ordered_subpics_dirs, policy) for nft in random_set if nft != random_set[index]]
            current_min_hamming = min(hamming_distances)
            for nft in metadata_list:
                hamming_distances = [unwrap_calc_hamming(random_nft, nft, ordered_subpics_dirs, policy) for random_nft in random_set if random_nft != random_set[index]]
                new_min_hamming = min(hamming_distances)
                if new_min_hamming > current_min_hamming:
                    print(f"Replacing at index {index} ({current_min_hamming} -> {new_min_hamming})")
                    random_set[index] = nft
                    break
    min_hamming = 1000
    for i in range(num_images):
        for j in range(i + 1, num_images):
            min_hamming = min(min_hamming, unwrap_calc_hamming(random_set[i], random_set[j], ordered_subpics_dirs, policy))
    random_set_ids = sorted([unwrap(nft, policy)['name'] for nft in random_set])
    print(f"Recommending the following for samples (min hamming: {min_hamming}) {random_set_ids}")
    for random_id in random_set_ids:
        png_filename = get_pic_filename(random_id)
        try:
            shutil.copy(os.path.join(pics_dir, png_filename), os.path.join(output_pics_dir, png_filename))
        except FileNotFoundError as e:
            print(e)

def compose_image(combination, in_pics_dir, ordered_subpics_dirs, linked_categories, output_size):
    background = Image.new('RGBA', output_size)
    for subpic_dir in ordered_subpics_dirs:
        if combination[subpic_dir]:
            filename = os.path.join(in_pics_dir, subpic_dir, combination[subpic_dir])
            if subpic_dir in linked_categories:
                linked_match = combination[linked_categories[subpic_dir]]
                root_dir = os.path.join(in_pics_dir, subpic_dir, linked_match)
                available_matches = os.listdir(root_dir)
                file_match = combination[subpic_dir] if f"{combination[subpic_dir]}.png" in available_matches else 'Default'
                #print(f"Looking for {combination[subpic_dir]} in subdir {root_dir} ({available_matches}), found {file_match}")
                filename = os.path.join(root_dir, file_match)
            foreground = Image.open(f"{filename}.png").convert('RGBA')
            background = Image.alpha_composite(background, foreground.resize(output_size))
    return background

def responsive_ordering_for(nft, responsive_ordering, ordered_subpics_dirs):
    for trait in nft:
        if nft[trait] in responsive_ordering:
            return responsive_ordering[nft[trait]]
    return ordered_subpics_dirs

def generate_image(nft, in_pics_dir, out_pics_dir, ordered_subpics_dirs, linked_categories, output_size, responsive_ordering):
    try:
        responsive_ordering_nft = responsive_ordering_for(nft, responsive_ordering, ordered_subpics_dirs)
        composed_image = compose_image(nft, in_pics_dir, responsive_ordering_nft, linked_categories, output_size)
        composed_image.save(os.path.join(out_pics_dir, get_pic_filename(nft['name'])))
    except FileNotFoundError as e:
        print(nft)
        print(e)

def calculate_hamming_distance(nft_a, nft_b, ordered_subpics_dirs):
    hamming_distance = 0
    for subpic_dir in ordered_subpics_dirs:
        if nft_a[subpic_dir] != nft_b[subpic_dir]:
            hamming_distance += 1
    return hamming_distance

def get_nft_name(prefix, i):
    return f"{prefix} {i+1}"

def get_pic_filename(nft_name):
    return f"{nft_name}.{FILE_FORMAT}"

def get_metadata_filename(nft_name):
    return f"{nft_name}.json"

def wrap(nft_name, values, policy):
    if not policy:
        return values
    return {
        "721": {
            policy: {
                nft_name: values
            },
            "version": "1.0"
        }
    }

def unwrap(metadata, policy):
    return next(iter(metadata["721"][policy].values())) if policy else metadata

def get_metadata_for(nft, policy, project):
    metadata = { "mediaType": "image/png", "project": project }
    for key in nft:
        metadata[key] = nft[key]
    return wrap(nft['name'], metadata, policy)

def write_metadata_statistics(full_list, metadata_statistics, expected_ratios, statistics_file, ordered_subpics_dirs):
    total_nfts = len(full_list)
    total_hamming_distance = 0
    min_hamming_distance = len(ordered_subpics_dirs)
    max_hamming_distance = 0
    for i in range(0, total_nfts):
        for j in range(i + 1, total_nfts):
            hamming_distance = calculate_hamming_distance(full_list[i], full_list[j], ordered_subpics_dirs)
            min_hamming_distance = min(min_hamming_distance, hamming_distance)
            max_hamming_distance = max(max_hamming_distance, hamming_distance)
            total_hamming_distance += hamming_distance
    avg_hamming_distance = total_hamming_distance / (total_nfts * (total_nfts + 1) / 2.0)
    statistics_file.write(f"Minimum hamming distance: {min_hamming_distance}\n")
    statistics_file.write(f"Maximum hamming distance: {max_hamming_distance}\n")
    statistics_file.write(f"Average hamming distance: {avg_hamming_distance}\n")
    statistics_file.write(f"\tNAME                               COUNT     ACTUAL      EXPECTED   VARIANCE\n")
    for subpic_dir in metadata_statistics.keys():
        statistics_file.write(f"{subpic_dir} -----------------\n")
        for item in metadata_statistics[subpic_dir].keys():
            total_number = metadata_statistics[subpic_dir][item]
            ratio = total_number / float(total_nfts) * 100.0
            expected = expected_ratios[subpic_dir][item] * 100.0 if item else 0.0
            variance = int(((ratio / expected) - 1) * 100) if expected else 1000.0
            statistics_file.write(f"\t{item:30}{total_number:10}{ratio:-10.1f}%{expected:-10.1f}%{variance:8}%\n")

def dump_metadata_files(full_list, metadata_dir, policy, project):
    for nft in full_list:
        with open(os.path.join(metadata_dir, get_metadata_filename(nft['name'])), 'w') as metadata_file:
            json.dump(get_metadata_for(nft, policy, project), metadata_file)

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

def is_illegal(nft_combination, excluded_combinations):
    traits = nft_combination.values()
    for excluder in excluded_combinations:
        for exclusion in excluded_combinations[excluder]:
            for trait in traits:
                if re.match(excluder, trait) and exclusion in traits:
                    #print(f"Excluding combination with '{excluder}' due to '{exclusion}'")
                    return True
    return False

def force_upgrade(nft_combination, forced_combinations):
    for trait in nft_combination:
        trait_val = nft_combination[trait]
        for forced_combination in forced_combinations:
            if re.match(forced_combination, trait_val):
                forced_traits = forced_combinations[forced_combination]
                for forced_trait in forced_traits:
                    nft_combination[forced_trait] = forced_traits[forced_trait]

def generate_nft_metadata(start_num, total_nfts, traits_ratios, acceptable_hamming, static_traits, variable_traits, ordered_subpics_dirs, prefix, excluded_combinations, forced_combinations, reference_set_dir):
    full_list = []
    last_printed = time.time()
    for i in range(start_num, start_num + total_nfts):
        nft_combination = {}
        while not nft_combination:
            if time.time() - last_printed > 5:
                print(f"Found {i - start_num} combinations so far...")
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
                if not nft_combination[subpic_dir] and subpic_dir in ordered_subpics_dirs:
                    print(f"Illegal state: required subpic {subpic_dir} not found (roll {dice_roll})")
            if is_illegal(nft_combination, excluded_combinations):
                nft_combination = {}
                continue
            force_upgrade(nft_combination, forced_combinations)
            for already_found in full_list + reference_set_dir:
                if calculate_hamming_distance(nft_combination, already_found, ordered_subpics_dirs) < acceptable_hamming:
                    #print(f"Repeating iteration {i}...")
                    nft_combination = {}
                    break
        nft_combination['name'] = get_nft_name(prefix, i)
        full_list.append(nft_combination)
    return full_list

def generate_nfts(start_num, total_nfts, traits_ratios, in_pics_dir, acceptable_hamming, output_pics_dir, metadata_dir, static_traits, variable_traits, ordered_subpics_dirs, linked_categories, output_size, policy, name_prefix, project, excluded_combinations, forced_combinations, reference_set_dir, responsive_ordering):
    nft_metadata = generate_nft_metadata(start_num, total_nfts, traits_ratios, acceptable_hamming, static_traits, variable_traits, ordered_subpics_dirs, name_prefix, excluded_combinations, forced_combinations, reference_set_dir)
    dump_metadata_files(nft_metadata, metadata_dir, policy, project) # Temporary in case something happens during the composition run
    metadata_statistics = compile_metadata_statistics(nft_metadata, variable_traits)
    with open(os.path.join(os.path.dirname(metadata_dir), 'metadata_statistics.tsv'), 'w') as statistics_file:
        write_metadata_statistics(nft_metadata, metadata_statistics, traits_ratios, statistics_file, ordered_subpics_dirs)
    for nft in nft_metadata:
        generate_image(nft, in_pics_dir, output_pics_dir, ordered_subpics_dirs, linked_categories, output_size, responsive_ordering)

def unwrap_calc_hamming(metadata_a, metadata_b, ordered_subpics_dirs, policy):
    return calculate_hamming_distance(unwrap(metadata_a, policy), unwrap(metadata_b, policy), ordered_subpics_dirs)

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

def check_uniqueness(nft_metadata, reference_set, ordered_subpics_dirs, policy):
    nft_paths = [os.path.join(nft_metadata, file) for file in os.listdir(nft_metadata)] if os.path.isdir(nft_metadata) else [nft_metadata]
    for nft_path in nft_paths:
        with open(nft_path, 'r') as nft_file:
            nft = json.load(nft_file)
        print(nft)
        hamming_distances = [(unwrap_calc_hamming(nft, reference_nft, ordered_subpics_dirs, policy), reference_nft) for reference_nft in reference_set]
        min_hamming = 10000
        for distance, ref_nft in hamming_distances:
            if distance < min_hamming:
                min_hamming = distance
                reference = ref_nft
        print(f"Min hamming found {min_hamming}:\n{reference}")

def generate_image_trait_filtered(nft, in_pics_dir, output_pics_dir, ordered_subpics_dirs, linked_categories, output_size, responsive_ordering, trait_filters):
    if not trait_filters:
        generate_image(nft, in_pics_dir, output_pics_dir, ordered_subpics_dirs, linked_categories, output_size, responsive_ordering)
    for trait_filter_opt in trait_filters:
        if trait_filter_opt[0] in nft and nft[trait_filter_opt[0]] == trait_filter_opt[1]:
            generate_image(nft, in_pics_dir, output_pics_dir, ordered_subpics_dirs, linked_categories, output_size, responsive_ordering)
            return

def regenerate_image(nft_metadata_filename, in_pics_dir, output_pics_dir, ordered_subpics_dirs, linked_categories, output_size, policy, trait_filter, responsive_ordering):
    with open(nft_metadata_filename, 'r') as nft_metadata_file:
        nfts_json = json.load(nft_metadata_file)
        nfts = nfts_json["721"][policy].values() if policy else [nfts_json]
        for nft in nfts:
            trait_filters = [trait_filter_opt.split('=') for trait_filter_opt in trait_filter.split(',')] if trait_filter else []
            generate_image_trait_filtered(nft, in_pics_dir, output_pics_dir, ordered_subpics_dirs, linked_categories, output_size, responsive_ordering, trait_filters)

def get_rarity_value(rarity_val):
    if isinstance(rarity_val, numbers.Number):
        return rarity_val
    return RARITY_MAPPING[rarity_val]

def read_generator_file(percentages_filename):
    percentages = {}
    with open(_args.percentages_file, 'r') as percentages_file:
        traits_json = json.load(percentages_file)
        ordered_categories = traits_json['ordered_categories']
        traits_descriptions = traits_json['rarity']
        for category in traits_descriptions:
            category_ratios = {}
            if not category in ordered_categories:
                raise ValueError(f"Unexpected category '{category}' found in percentages file")
            category_traits = traits_descriptions[category]
            category_total = 0.0
            for attr in category_traits:
                category_total += get_rarity_value(category_traits[attr])
            percentages[category] = {}
            for attr in category_traits:
                percentages[category][attr] = get_rarity_value(category_traits[attr]) / category_total
    return {
        'forced_combinations': traits_json['forced_combinations'],
        'excluded_combinations': traits_json['excluded_combinations'],
        'linked_categories': traits_json['linked_categories'],
        'responsive_ordering': traits_json['responsive_ordering'],
        'ordered_categories': ordered_categories,
        'rarity': percentages
    }

def make_and_get_output_dir(prefix):
    output_dir = os.path.join(prefix, str(int(time.time())))
    metadata_dir = os.path.join(output_dir, 'metadata')
    pics_dir = os.path.join(output_dir, 'pics')
    [os.makedirs(out_dir) for out_dir in [output_dir, metadata_dir, pics_dir]]
    return (pics_dir, metadata_dir)

def get_parser():
    parser = argparse.ArgumentParser(description='Generate NFTs')
    subparsers = parser.add_subparsers(dest='command')

    nfts_parser = subparsers.add_parser('generate-nfts', help='Compose NFTs according to trait gradient algorithm')
    nfts_parser.add_argument('--output-dir', type=str, required=True)
    nfts_parser.add_argument('--total-nfts', type=int, required=True)
    nfts_parser.add_argument('--policy', type=str, required=False)
    nfts_parser.add_argument('--project', type=str, required=True)
    nfts_parser.add_argument('--name-prefix', type=str, required=True)
    nfts_parser.add_argument('--dimension', type=int, required=True)
    nfts_parser.add_argument('--percentages-file', required=True)
    nfts_parser.add_argument('--min-hamming', type=int, required=True)
    nfts_parser.add_argument('--pics-dir', required=True)
    nfts_parser.add_argument("--static-trait", action='append', type=lambda kv: kv.split("="), dest='static_traits')
    nfts_parser.add_argument('--start-num', type=int, required=False, default=0)
    nfts_parser.add_argument('--reference-set-dir', type=str, required=False)

    samples_parser = subparsers.add_parser('generate-samples', help='Generate recommendations for unique samples')
    samples_parser.add_argument('--output-dir', type=str, required=True)
    samples_parser.add_argument('--metadata-dir', required=True)
    samples_parser.add_argument('--pics-dir', required=True)
    samples_parser.add_argument('--num-images', type=int, required=True)
    samples_parser.add_argument('--num-iterations', type=int, required=True)
    samples_parser.add_argument('--policy', type=str, required=False)

    onchain_parser = subparsers.add_parser('on-chain', help='Create metadata with PNG on chain (size checks not enforced, must be <16KB)')
    onchain_parser.add_argument('--output-dir', type=str, required=True)
    onchain_parser.add_argument('--metadata-dir', required=True)
    onchain_parser.add_argument('--pics-dir', required=True)
    onchain_parser.add_argument('--policy', required=False)
    onchain_parser.add_argument('--project', required=True)

    ipfs_parser = subparsers.add_parser('upload-to-ipfs', help='Upload raw images to IPFS')
    ipfs_parser.add_argument('--output-dir', type=str, required=True)
    ipfs_parser.add_argument('--metadata-dir', required=True)
    ipfs_parser.add_argument('--pics-dir', required=True)
    ipfs_parser.add_argument('--policy', type=str, required=False)
    ipfs_parser.add_argument('--project', type=str, required=True)
    ipfs_parser.add_argument('--nft-storage-key', type=str, required=True)
    ipfs_parser.add_argument('--existing-uploads-dir', type=str, required=False)
    ipfs_parser.add_argument('--cip-25', action='store_true')

    uniq_parser = subparsers.add_parser('check-uniqueness', help='Compare the uniqueness of this NFT to the reference set')
    uniq_parser.add_argument('--nft-metadata', required=True)
    uniq_parser.add_argument('--reference-set-dir', required=True)
    uniq_parser.add_argument('--percentages-file', type=str, required=True)
    uniq_parser.add_argument('--policy', type=str, required=False)

    regen_parser = subparsers.add_parser('regenerate-image', help='Regenerate an image based on metadata file provided (useful for 1-of-1s)')
    regen_parser.add_argument('--output-dir', type=str, required=True)
    regen_parser.add_argument('--percentages-file', type=str, required=True)
    regen_parser.add_argument('--nft-metadata-file', required=True)
    regen_parser.add_argument('--dimension', type=int, required=True)
    regen_parser.add_argument('--pics-dir', required=True)
    regen_parser.add_argument('--policy', required=False)
    regen_parser.add_argument('--trait-filter', required=False)

    regen_dir_parser = subparsers.add_parser('regenerate-directory', help='Regenerate a directory based on metadata files provided (useful for upgraded art)')
    regen_dir_parser.add_argument('--output-dir', type=str, required=True)
    regen_dir_parser.add_argument('--percentages-file', type=str, required=True)
    regen_dir_parser.add_argument('--nft-directory', required=True)
    regen_dir_parser.add_argument('--dimension', type=int, required=True)
    regen_dir_parser.add_argument('--pics-dir', required=True)
    regen_dir_parser.add_argument('--no-unwrap', action='store_true')
    regen_dir_parser.add_argument('--policy', required=False)
    regen_dir_parser.add_argument('--trait-filter', required=False)

    return parser

def seed_random():
    random.seed(12345)

if __name__ == '__main__':
    seed_random()
    _args = get_parser().parse_args()
    if not _args.command in ['check-uniqueness']:
        _pics_dir, _metadata_dir = make_and_get_output_dir(_args.output_dir)
    if _args.command == 'generate-nfts':
        _traits_rarity = read_generator_file(_args.percentages_file)
        _excluded_combinations = _traits_rarity['excluded_combinations']
        _forced_combinations = _traits_rarity['forced_combinations']
        _linked_categories = _traits_rarity['linked_categories']
        _ordered_subpics_dir = _traits_rarity['ordered_categories']
        _responsive_ordering = _traits_rarity['responsive_ordering']
        _traits_ratios = _traits_rarity['rarity']
        _static_traits = dict(_args.static_traits) if _args.static_traits else {}
        _variable_traits = [trait for trait in _ordered_subpics_dir if not (trait in _static_traits.keys())]
        print(f"Creating {_args.total_nfts} NFT w/hamming >{_args.min_hamming} & ratios from '{_args.percentages_file}' to dir '{_pics_dir}'")
        print(f"Static traits specified: {_static_traits}")
        print(f"Variable traits specified: {_variable_traits}")
        _reference_set = load_metadata(_args.reference_set_dir) if _args.reference_set_dir else []
        generate_nfts(_args.start_num, _args.total_nfts, _traits_ratios, _args.pics_dir, _args.min_hamming, _pics_dir, _metadata_dir, _static_traits, _variable_traits, _ordered_subpics_dir, _linked_categories, (_args.dimension, _args.dimension), _args.policy, _args.name_prefix, _args.project, _excluded_combinations, _forced_combinations, _reference_set, _responsive_ordering)
    elif _args.command == 'generate-samples':
        generate_sample_images(_args.metadata_dir, _args.pics_dir, _args.num_images, _args.num_iterations, _pics_dir, _ordered_subpics_dir, _args.policy)
    elif _args.command == 'on-chain':
        generate_new_metadata_on_chain(_args.metadata_dir, _args.pics_dir, _metadata_dir, _args.policy, _args.project)
    elif _args.command == 'upload-to-ipfs':
        upload_to_ipfs_new_metadata(_args.pics_dir, _args.metadata_dir, _metadata_dir, _args.policy, _args.project, _args.nft_storage_key, _args.existing_uploads_dir, _args.cip_25)
    elif _args.command == 'check-uniqueness':
        _traits_rarity = read_generator_file(_args.percentages_file)
        _ordered_subpics_dir = _traits_rarity['ordered_categories']
        _reference_set = load_metadata(_args.reference_set_dir)
        check_uniqueness(_args.nft_metadata, _reference_set, _ordered_subpics_dir, _args.policy)
    elif _args.command == 'regenerate-directory':
        _traits_rarity = read_generator_file(_args.percentages_file)
        _ordered_subpics_dir = _traits_rarity['ordered_categories']
        _linked_categories = _traits_rarity['linked_categories']
        _responsive_ordering = _traits_rarity['responsive_ordering']
        for nft_metadata_file in os.listdir(_args.nft_directory):
            nft_metadata_path = os.path.join(_args.nft_directory, nft_metadata_file)
            regenerate_image(nft_metadata_path, _args.pics_dir, _pics_dir, _ordered_subpics_dir, _linked_categories, (_args.dimension, _args.dimension), _args.policy, _args.trait_filter, _responsive_ordering)
    elif _args.command == 'regenerate-image':
        _traits_rarity = read_generator_file(_args.percentages_file)
        _ordered_subpics_dir = _traits_rarity['ordered_categories']
        _linked_categories = _traits_rarity['linked_categories']
        _responsive_ordering = _traits_rarity['responsive_ordering']
        regenerate_image(_args.nft_metadata_file, _args.pics_dir, _pics_dir, _ordered_subpics_dir, _linked_categories, (_args.dimension, _args.dimension), _args.policy, _args.trait_filter, _responsive_ordering)
    else:
        raise ValueError("No command passed to the program.  Use -h for help.")
