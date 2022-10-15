import editdistance

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    target_text = target_text.split(' ')
    if len(target_text) == 0:
        return 1
    predicted_text = predicted_text.split(' ')
    return editdistance.distance(predicted_text, target_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1
    return editdistance.distance(predicted_text, target_text) / len(target_text)
