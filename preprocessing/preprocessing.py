import re
from collections import Counter
from langdetect import detect
from gensim.summarization import summarize
import pandas as pd

def numbered_reference_removal(text):
    # refs are typically in the form [n] in the text.
    matches = re.findall(r'\[[0-9]+?\]', text)
    counter = Counter(matches)
    for match in matches:
        if counter[match] != 2:
            # print(f"Didn't find a reference twice in the text, but {counter[match]} times. Cannot remove.")
            return text
    
    assert len(matches)%2 == 0
    if len(matches) == 0:
        return text
    N = len(matches)//2
    is_ordered = True
    n = 0
    for s in matches:
        is_ordered = (s == f'[{n%N+1}]')
        if not is_ordered : 
            # print(f"Not ordered, {str(matches)}")
            return text
        n += 1
    
    # Remove all references after [1]
    res = re.findall(
        r'^.*?\[1\].+?\[1\]',
        text
    )
    return res[0]

def reference_removal_en(text):
    res = re.sub(r'(?i)see also.*', '', text)
    res = re.sub(r'References.*', '', res)
    res = re.sub("\[[0-9]+\]", "", res)
    return res

# Before dealing with N/A because otherwise, this edit would just undo the other.
# We can use this to remove the "TRANSCRIPT" and "SPEECH" tags.
# We also notice that most of the time, the content is preceded by a date.
# It would be useful to make a date parser to remove the first chunk.

def first_date_extractor(text):
    if len(text) > 0:
        res = re.sub('^(.*?)[1-9][0-9]* (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|'
                     + 'Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(Nov|Dec)(?:ember)?) (19|20)[0-9][0-9]',
                     '', text)
        return res

def remove_title(x):
    if not x["text"] is None and isinstance(x["text"], str):
        res = re.sub(x["title"], '', x["text"]).strip()
        return res
    else:
        return None

def website_remover(text):
    # Remove websites
    regex = "((http|https)://)(www.)?" \
        + "[a-zA-Z0-9@:%._\\+~#?&//=]{2,256}\\.[a-z]" \
        + "{2,6}\\b([-a-zA-Z0-9@:%._\\+~#?&//=]*)"
    res = re.sub(regex, "", text).strip()
    return res

def tag_removal(text):
    # Remove tags
    res = re.sub('SPEECH', '', text)
    res = re.sub('TRANSCRIPT', '', res)
    res = re.sub("Introduction", "", res)
    res = re.sub("Summary", "", res)
    return res


def summarizeLine(text, tolist=False):
    try:
        res = summarize(text, word_count=500)
        if tolist:
            res = res.split("\n")
    except:
        return text
    return res

def find_footnote(x):
    found = re.match(r"(.*)footnote", x, re.IGNORECASE)
    if not found is None:
        insensitive_footnote = re.compile(re.escape('footnote'), re.IGNORECASE)
        return insensitive_footnote.sub("", found.group()).strip()
    else:
        return x

def find_useless_thanks(x):
    if x is not None:
        found = re.findall(r"([^.]*?(thank | congratulat)[^.]*\.)", x, re.IGNORECASE)
    if not found is None:
        res = x
        for substring in found:
            res = re.sub(re.escape(substring[0]), "", res)
        return res
    return x

def remove_video_code(text):
    if not text is None:
        res = re.sub("Accessible Keys for Video.*myPlayer\.play\(\);(.*?)\}(.*?)\}", "", text).strip()
        res = re.sub("^(Watch|View) Video", "", res)
        return res.strip()
    return text

def remove_refs_fed(text):
    if not text is None:
        res = re.sub(r'References.*', '', text)
        res = re.sub(r'1\.(.+?)Return to text.*', '', res)
        res = re.sub(r'Return to text.*', '', res)
        return res.strip()
    return text

def remove_greetings(text):
    if not text is None:
        res = re.sub(r'^(.*?)Good (morning|afternoon|evening)[^.]*\.', '', text)
        res = re.sub(r'^(.*?)Ladies and (g|G)entlemen[^.]*\.', '', res)
        res = re.sub(r'Hello.', '', res)
        return res.strip()
    return text

def pipeline_en(x, tolist=False):
    res = remove_title(x)
    if res is None:
        return x["title"]
    res = numbered_reference_removal(res)
    res = remove_greetings(res)
    res = reference_removal_en(res)
    res = tag_removal(res).strip()
    res = first_date_extractor(res).strip()
    res = summarizeLine(res, tolist)

    return res

def pipeline_fed(fed):
    res = fed["text"].apply(find_footnote)
    res = res.apply(remove_video_code)
    res = res.apply(find_useless_thanks)
    res = res.apply(remove_refs_fed)
    res = res.apply(remove_greetings)
    res = res.apply(website_remover)
    fed["text_"] = res
    return fed

def fast_detect(x, bound=500):
    return detect(x[:min(len(x), bound)])


def main():
    FILENAME_ECB = "../data/ecb_data.csv"
    FILENAME_FED = "../data/fed_data.csv"

    ecb = pd.read_csv(FILENAME_ECB, index_col=0)
    fed = pd.read_csv(FILENAME_FED, index_col=0)

    ecb["text_"] = ecb.apply(pipeline_en, axis=1)
    ecb["lang"] = ecb["text_"].apply(fast_detect)
    fed["lang"] = fed["text"].apply(fast_detect)

    with open("../data/ecb_data_preprocessed.csv", "w+", encoding="utf-8") as f:
        ecb.to_csv(f)

    with open("../data/fed_data_preprocessed.csv", "w+", encoding="utf-8") as f:
        fed.to_csv(f)

    print("Finished preprocessing.")

if __name__=="__main__":
    main()
