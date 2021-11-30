def generate_tags(sample, tokenizer, max_length=75):
    """
    Generate tags for each token in the text
    """
    tags = []
    aspect_idx = 0
    started = False
    all_aspects_covered = False

    text, aspects = sample['text'], sample['aspects']
    aspects = sorted(aspects, key=lambda x: x['from'])

    tokens = tokenizer.tokenize(text, add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True)
    tokenized = tokenizer(text, padding='max_length', truncation=True, return_offsets_mapping=True)

    for token, offset in zip(tokens, tokenized['offset_mapping']):

        from_ = offset[0]; to = offset[1]
        if from_ == to == 0:
            tag = 'Q'  # represents the ignore tag for special tokens

        elif all_aspects_covered or to < aspects[aspect_idx]['from']: # either all aspects are covered or the current token is before the current aspect
            tag= 'O' # represents the tag for non-aspect tokens
            started = False

        elif from_ >= aspects[aspect_idx]['from'] and to <= aspects[aspect_idx]['to']: # If the current token boundaries are within the aspect boundaries
            if not started:
                started = True
                tag = 'B'
            else:
                if token.startswith("##"):
                    tag = 'X' # represents the tag for subtokens
                else:
                    tag = 'I' # represents the tag for continuing aspect

        if not all_aspects_covered and to >= aspects[aspect_idx]['to']: # If the current token boundaries are after the aspect boundaries
            aspect_idx += 1 # move to the next aspect
            if aspect_idx >= len(aspects):
                all_aspects_covered = True
            started = False

        tags.append(tag)

    return tags, tokens, tokenized