import openai
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import argparse
from pprint import pprint

# set openAI key in separate .env file w/ content
# OPENAIKEY = yourkey
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

items = pd.read_csv('data_profiled_cn.csv', sep=';')

instructions = "Your task is to take the following counternarrative and make it more personalized for a person of demographics described below.\n\n"
cn_prefix = "Counternarrative: "
demo_prefix = "Target of counternarrative: "
example = """EXAMPLE:

Counternarrative: I hate to break it to you, but women are not cunts. Feminists and humanists are fighting for the same rights that they hope to have in a thousand years. Hating women is the problem, not the solution.

Target of counternarrative: Older man.

Personalized counternarrative: I understand that you may have grown up in a different time when attitudes towards women were different, but it's important to recognize that language like "cunt" is not acceptable and only serves to perpetuate harmful stereotypes and attitudes towards women.
Feminists and humanists are not trying to take away your rights or make you feel inferior; rather, they are fighting for equality and the recognition that women should have the same rights and opportunities as men.
This fight for equality is not new, and it is not going away anytime soon.

Counternarrative: Muslim rape our girls.

Target of counternarrative: Younger woman

Personalized counternarrative: I understand that the media can be overwhelming and that you may feel frustrated or angry about the topic of rape and religion. It can be difficult to sort through the rumors and misinformation.
It's important to approach these issues with a critical eye and an open mind, and to recognize that not all Muslims are violent or abusive.The problem of violence is not unique to any one religion or culture.
It affects people of all backgrounds and faiths, and we should work together to address it in a constructive and compassionate way.

"""

def get_personalized_cn(
        cn,
        demo,
        instructions,
        cn_prefix,
        demo_prefix,
        model_name,
        example="",
        use_example=False,
    ):
    if use_example:
        input = instructions + example + cn_prefix + cn + "\n\n" + demo_prefix + demo + "\n\n" + "Personalized counternarrative: "
    else:
        input = instructions + cn_prefix + cn + "\n\n" + demo_prefix + demo + "\n\n" + "Personalized counternarrative: "

    # get completion from GPT-3
    response = openai.Completion.create(
        engine      = model_name,
        prompt      = input,
        max_tokens  = 128,
        temperature = 1, #decides level of creativity (0-1)
        logprobs    = 0,
        echo        = True,
        n           = 1, #number of counternarratives per item
    )
    pprint(response)
    # get only the generated continuation
    text_offsets = response.choices[0]['logprobs']['text_offset']
    cutIndex = text_offsets.index(max(i for i in text_offsets if i < len(input))) + 1
    endIndex = response.usage.total_tokens
    personalized_cn = response.choices[0]["logprobs"]["tokens"][cutIndex:endIndex]

    return personalized_cn

# output dataframe
out_df = items[:5].copy()
personalized_cns = []
for i, row in items[:5].iterrows():
    cn = row['cn']
    age = 'older' if row['age'] == 25 else 'younger'
    gender = 'man ' if row['gender'] == 'm' else 'woman '
    demo = 'for a ' + gender + age
    personalized_cn = get_personalized_cn(cn, demo, instructions, cn_prefix, demo_prefix, model_name="text-davinci-003", use_example=False)
    print("Personalized cn ", personalized_cn)
    personalized_cn = ''.join(personalized_cn).replace('\n', '')
    print("Personalized cn formatted ", personalized_cn)
    personalized_cns.append(personalized_cn)

out_df['personalized_cn'] = personalized_cns

out_df.to_csv("personalized_cn_first5.csv", sep=';', index=False)
