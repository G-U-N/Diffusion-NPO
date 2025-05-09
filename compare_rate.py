import os
import random
from PIL import Image
import numpy as np
import utils.aes_utils as aes_utils
import utils.hps_utils as hps_utils
import utils.pickscore_utils as pickscore_utils
import ImageReward as RM


aes_selector = aes_utils.Selector("cuda")
hps_selector = hps_utils.Selector("cuda")
pickscore_selector = pickscore_utils.Selector("cuda")
image_reward_model = RM.load("ImageReward-v1.0", "cuda")


def select_common_files(folder_path1, folder_path2, num_files=500):
    all_files1 = {f for f in os.listdir(folder_path1) if f.endswith(".png")}
    all_files2 = {f for f in os.listdir(folder_path2) if f.endswith(".png")}
    common_files = list(all_files1.intersection(all_files2))
    selected_files = random.sample(common_files, min(num_files, len(common_files)))
    return selected_files


def aes_score(selector, folder_path, selected_files):
    scores = []
    for filename in selected_files:
        text_filename = filename.replace(".png", ".txt")
        if text_filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            text_path = os.path.join(folder_path, text_filename)
            with open(text_path, "r") as f:
                text = f.read().strip()
            score = selector.score(Image.open(image_path), text)
            scores.append(score[0])
    return scores


def hps_score(selector, folder_path, selected_files):
    scores = []
    for filename in selected_files:
        text_filename = filename.replace(".png", ".txt")
        if text_filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            text_path = os.path.join(folder_path, text_filename)
            with open(text_path, "r") as f:
                text = f.read().strip()
            score = selector.score(image_path, text)
            scores.append(score[0])
    return scores


def image_reward_score(model, folder_path, selected_files):
    scores = []
    for filename in selected_files:
        text_filename = filename.replace(".png", ".txt")
        if text_filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            text_path = os.path.join(folder_path, text_filename)
            with open(text_path, "r") as f:
                text = f.read().strip()
            score = model.score(text, [image_path])
            scores.append(score)
    return scores


def pickscore_score(selector, folder_path, selected_files):
    scores = []
    for filename in selected_files:
        text_filename = filename.replace(".png", ".txt")
        if text_filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            text_path = os.path.join(folder_path, text_filename)
            with open(text_path, "r") as f:
                text = f.read().strip()
            score = selector.score(Image.open(image_path), text)
            scores.append(score[0])
    return scores


def calculate_win_probability(scores1, scores2):
    wins1 = sum(1 for s1, s2 in zip(scores1, scores2) if s1 > s2)
    wins2 = sum(1 for s1, s2 in zip(scores1, scores2) if s1 < s2)
    total = len(scores1)
    return wins1 / total, wins2 / total


def main(folder_path1, folder_path2):
    # Load models and selectors

    # Select common files
    selected_files = select_common_files(folder_path1, folder_path2)

    # Compute scores for folder 1
    aes_scores1 = aes_score(aes_selector, folder_path1, selected_files)
    hps_scores1 = hps_score(hps_selector, folder_path1, selected_files)
    image_reward_scores1 = image_reward_score(
        image_reward_model, folder_path1, selected_files
    )
    pickscore_scores1 = pickscore_score(
        pickscore_selector, folder_path1, selected_files
    )

    # Compute scores for folder 2
    aes_scores2 = aes_score(aes_selector, folder_path2, selected_files)
    hps_scores2 = hps_score(hps_selector, folder_path2, selected_files)
    image_reward_scores2 = image_reward_score(
        image_reward_model, folder_path2, selected_files
    )
    pickscore_scores2 = pickscore_score(
        pickscore_selector, folder_path2, selected_files
    )

    # Calculate win probabilities
    aes_win_prob1, aes_win_prob2 = calculate_win_probability(aes_scores1, aes_scores2)
    hps_win_prob1, hps_win_prob2 = calculate_win_probability(hps_scores1, hps_scores2)
    image_reward_win_prob1, image_reward_win_prob2 = calculate_win_probability(
        image_reward_scores1, image_reward_scores2
    )
    pickscore_win_prob1, pickscore_win_prob2 = calculate_win_probability(
        pickscore_scores1, pickscore_scores2
    )

    # Print results
    print(f"Comparing {folder_path1} and {folder_path2}:")
    print(
        f"AES win probability: Folder 1 = {aes_win_prob1}, Folder 2 = {aes_win_prob2}"
    )
    print(
        f"HPS win probability: Folder 1 = {hps_win_prob1}, Folder 2 = {hps_win_prob2}"
    )
    print(
        f"ImageReward win probability: Folder 1 = {image_reward_win_prob1}, Folder 2 = {image_reward_win_prob2}"
    )
    print(
        f"PickScore win probability: Folder 1 = {pickscore_win_prob1}, Folder 2 = {pickscore_win_prob2}"
    )

    # Calculate mean scores for folder 1
    aes_mean_score1 = np.mean(aes_scores1)
    hps_mean_score1 = np.mean(hps_scores1)
    image_reward_mean_score1 = np.mean(image_reward_scores1)
    pickscore_mean_score1 = np.mean(pickscore_scores1)

    # Calculate mean scores for folder 2
    aes_mean_score2 = np.mean(aes_scores2)
    hps_mean_score2 = np.mean(hps_scores2)
    image_reward_mean_score2 = np.mean(image_reward_scores2)
    pickscore_mean_score2 = np.mean(pickscore_scores2)

    print(f"AES mean score: Folder 1 = {aes_mean_score1}, Folder 2 = {aes_mean_score2}")
    print(f"HPS mean score: Folder 1 = {hps_mean_score1}, Folder 2 = {hps_mean_score2}")
    print(
        f"ImageReward mean score: Folder 1 = {image_reward_mean_score1}, Folder 2 = {image_reward_mean_score2}"
    )
    print(
        f"PickScore mean score: Folder 1 = {pickscore_mean_score1}, Folder 2 = {pickscore_mean_score2}"
    )


if __name__ == "__main__":
    folder_pairs = [
        ("[FOLDER 1][ADD the FOLDER PATH HERE]", "[FOLDER 2][ADD the FOLDER PATH HERE]"),
    ]
    for folder_path1, folder_path2 in folder_pairs:
        print(folder_path1)
        print(folder_path2)
        main(folder_path1, folder_path2)
