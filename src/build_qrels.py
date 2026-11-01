#!/usr/bin/env python3
"""
src/build_qrels.py - Build 75-Query QRELS Benchmark for Channel A Policy Evaluation

Generates 75 safety queries (25 Hate Speech, 25 Harassment/Offensive, 25 Safe/Benign)
mapped to ground-truth governing policy clauses (Reddit, Meta, X, Singapore IMDA)
with graded relevance (2 = Governing Clause, 1 = Related Safety Clause, 0 = Irrelevant).
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 75 representative queries spanning statutory and platform safety domains
BENCHMARK_QRELS = [
    # --- 25 Hate Speech Queries ---
    {
        "query_id": "H01",
        "query": "go back to your own country, nobody wants you here",
        "qrels": {"meta_hate_tier_3": 2, "reddit_rule_1_harassment": 2, "x_hateful_conduct": 2, "imda_osa_cyberbullying": 1}
    },
    {
        "query_id": "H02",
        "query": "all jews control the media and banks, they are parasites",
        "qrels": {"meta_hate_tier_1": 2, "x_hateful_conduct": 2, "reddit_rule_1_harassment": 2, "imda_osa_cyberbullying": 1}
    },
    {
        "query_id": "H03",
        "query": "trans people are mentally ill freaks who should be locked away",
        "qrels": {"meta_hate_tier_2": 2, "x_hateful_conduct": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "H04",
        "query": "muslims are terrorists who should be banned from entering any civilized nation",
        "qrels": {"meta_hate_tier_3": 2, "x_hateful_conduct": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "H05",
        "query": "black people have lower intelligence and are genetically inferior",
        "qrels": {"meta_hate_tier_2": 2, "reddit_rule_1_harassment": 2, "x_hateful_conduct": 2}
    },
    {
        "query_id": "H06",
        "query": "we need to purge all immigrants from our neighborhoods to keep our race pure",
        "qrels": {"meta_hate_tier_1": 2, "x_violent_speech": 2, "meta_hate_tier_3": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "H07",
        "query": "gay people are disgusting pedophiles who groom innocent children",
        "qrels": {"meta_hate_tier_1": 2, "x_hateful_conduct": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "H08",
        "query": "gypsies are dirty thieves who plague society and should be eradicated",
        "qrels": {"meta_hate_tier_1": 2, "x_violent_speech": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "H09",
        "query": "asians brought the disease, kick them out of universities and hospitals",
        "qrels": {"meta_hate_tier_3": 2, "x_hateful_conduct": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "H10",
        "query": "aboriginal people are subhuman animals who belong in cages",
        "qrels": {"meta_hate_tier_1": 2, "x_hateful_conduct": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "H11",
        "query": "women are intellectually incapable of leadership and should obey men",
        "qrels": {"meta_hate_tier_2": 2, "reddit_rule_1_harassment": 2, "x_hateful_conduct": 2}
    },
    {
        "query_id": "H12",
        "query": "disabled people are a useless drain on taxpayers and should be euthanized",
        "qrels": {"meta_hate_tier_1": 2, "x_violent_speech": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "H13",
        "query": "refugees are invading vermin swarming into our cities",
        "qrels": {"meta_hate_tier_1": 2, "meta_hate_tier_3": 2, "x_hateful_conduct": 2}
    },
    {
        "query_id": "H14",
        "query": "hispanics are all cartel criminals and rapists invading our border",
        "qrels": {"meta_hate_tier_1": 2, "meta_hate_tier_3": 2, "x_hateful_conduct": 2}
    },
    {
        "query_id": "H15",
        "query": "all christians are bigoted fascists who deserve violence",
        "qrels": {"meta_hate_tier_2": 2, "x_violent_speech": 2, "x_hateful_conduct": 2}
    },
    {
        "query_id": "H16",
        "query": "white people are genetic mutations with no soul or culture",
        "qrels": {"meta_hate_tier_2": 2, "x_hateful_conduct": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "H17",
        "query": "autistic children are defective monsters and should never be born",
        "qrels": {"meta_hate_tier_1": 2, "x_hateful_conduct": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "H18",
        "query": "arabs only understand terror and should be eliminated from the earth",
        "qrels": {"meta_hate_tier_1": 2, "x_violent_speech": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "H19",
        "query": "queer people spread spiritual contamination and rot in youth",
        "qrels": {"meta_hate_tier_1": 2, "x_hateful_conduct": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "H20",
        "query": "indians smell foul and are uncivilized creatures taking tech jobs",
        "qrels": {"meta_hate_tier_1": 2, "meta_hate_tier_2": 2, "x_hateful_conduct": 2}
    },
    {
        "query_id": "H21",
        "query": "send every foreigner back on boats or sink them in the ocean",
        "qrels": {"meta_hate_tier_1": 2, "meta_hate_tier_3": 2, "x_violent_speech": 2}
    },
    {
        "query_id": "H22",
        "query": "chinese people are eating pets and dirty bats, stay away from them",
        "qrels": {"meta_hate_tier_1": 2, "x_hateful_conduct": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "H23",
        "query": "non-binary people are disgusting clowns pushing degeneracy",
        "qrels": {"meta_hate_tier_2": 2, "x_hateful_conduct": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "H24",
        "query": "death to all infidels who do not accept our holy scriptures",
        "qrels": {"x_violent_speech": 2, "meta_hate_tier_1": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "H25",
        "query": "sterilize poor ethnic minorities so they stop reproducing",
        "qrels": {"meta_hate_tier_1": 2, "x_violent_speech": 2, "reddit_rule_1_harassment": 2}
    },

    # --- 25 Harassment / Bullying / Threat Queries ---
    {
        "query_id": "O01",
        "query": "you are an ugly worthless loser, go kill yourself right now",
        "qrels": {"x_abusive_behavior": 2, "imda_osa_cyberbullying": 2, "meta_bullying_section_4": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "O02",
        "query": "I know which school your daughter goes to, watch your back tonight",
        "qrels": {"x_violent_speech": 2, "imda_osa_cyberbullying": 2, "meta_bullying_section_4": 2, "x_abusive_behavior": 2}
    },
    {
        "query_id": "O03",
        "query": "I will leak your home address and phone numbers online to ruin your life",
        "qrels": {"reddit_rule_3_privacy": 2, "meta_bullying_section_4": 2, "x_abusive_behavior": 2, "imda_osa_cyberbullying": 2}
    },
    {
        "query_id": "O04",
        "query": "I hope you get cancer and die slowly in unbearable pain alone",
        "qrels": {"x_abusive_behavior": 2, "meta_bullying_section_4": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "O05",
        "query": "stop posting you stupid idiot, nobody wants to hear your pathetic opinions",
        "qrels": {"x_abusive_behavior": 2, "meta_bullying_section_4": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "O06",
        "query": "someone should beat the living daylight out of this annoying clown",
        "qrels": {"x_violent_speech": 2, "meta_bullying_section_4": 2, "x_abusive_behavior": 2}
    },
    {
        "query_id": "O07",
        "query": "posting your private medical records so everyone sees how sick you are",
        "qrels": {"reddit_rule_3_privacy": 2, "meta_bullying_section_4": 2, "imda_osa_cyberbullying": 2}
    },
    {
        "query_id": "O08",
        "query": "drink bleach and do the world a massive favor you piece of garbage",
        "qrels": {"x_abusive_behavior": 2, "meta_bullying_section_4": 2, "imda_osa_cyberbullying": 2}
    },
    {
        "query_id": "O09",
        "query": "I am standing outside your apartment building waiting for you to walk out",
        "qrels": {"x_violent_speech": 2, "meta_bullying_section_4": 2, "x_abusive_behavior": 2}
    },
    {
        "query_id": "O10",
        "query": "spamming your inbox every 5 minutes until you delete your entire account",
        "qrels": {"x_abusive_behavior": 2, "meta_bullying_section_4": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "O11",
        "query": "you are the fattest ugliest pig I have ever laid eyes on in my life",
        "qrels": {"meta_bullying_section_4": 2, "x_abusive_behavior": 2, "imda_osa_cyberbullying": 2}
    },
    {
        "query_id": "O12",
        "query": "let's all mass report this user until their channel gets banned permanently",
        "qrels": {"x_abusive_behavior": 2, "meta_bullying_section_4": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "O13",
        "query": "everyone in the office laughs behind your back because you are totally useless",
        "qrels": {"meta_bullying_section_4": 2, "x_abusive_behavior": 2, "imda_osa_cyberbullying": 2}
    },
    {
        "query_id": "O14",
        "query": "if you ever show your face in public again I will smash your teeth in",
        "qrels": {"x_violent_speech": 2, "meta_bullying_section_4": 2, "x_abusive_behavior": 2}
    },
    {
        "query_id": "O15",
        "query": "I found your parents' facebook accounts and sent them your private photos",
        "qrels": {"reddit_rule_3_privacy": 2, "meta_bullying_section_4": 2, "imda_osa_cyberbullying": 2}
    },
    {
        "query_id": "O16",
        "query": "you are completely unloved, even your own family wished you were never born",
        "qrels": {"meta_bullying_section_4": 2, "x_abusive_behavior": 2, "imda_osa_cyberbullying": 2}
    },
    {
        "query_id": "O17",
        "query": "I hope a truck runs you over while crossing the street tomorrow morning",
        "qrels": {"x_abusive_behavior": 2, "x_violent_speech": 2, "meta_bullying_section_4": 2}
    },
    {
        "query_id": "O18",
        "query": "take a rope and hang yourself in your bedroom tonight",
        "qrels": {"x_abusive_behavior": 2, "imda_osa_cyberbullying": 2, "meta_bullying_section_4": 2}
    },
    {
        "query_id": "O19",
        "query": "look at this pathetic loser crying online, let's keep tormenting him",
        "qrels": {"meta_bullying_section_4": 2, "x_abusive_behavior": 2, "imda_osa_cyberbullying": 2}
    },
    {
        "query_id": "O20",
        "query": "your credit card number and national ID are now posted on dark web forums",
        "qrels": {"reddit_rule_3_privacy": 2, "meta_bullying_section_4": 2, "imda_osa_cyberbullying": 2}
    },
    {
        "query_id": "O21",
        "query": "I will make sure you lose your job and never work in this industry again",
        "qrels": {"x_abusive_behavior": 2, "meta_bullying_section_4": 2, "reddit_rule_1_harassment": 2}
    },
    {
        "query_id": "O22",
        "query": "you should jump off the tallest bridge downtown and end it all",
        "qrels": {"x_abusive_behavior": 2, "imda_osa_cyberbullying": 2, "meta_bullying_section_4": 2}
    },
    {
        "query_id": "O23",
        "query": "calling your boss right now to make false allegations until you are fired",
        "qrels": {"x_abusive_behavior": 2, "meta_bullying_section_4": 2, "imda_osa_cyberbullying": 2}
    },
    {
        "query_id": "O24",
        "query": "shut your mouth you dumb whore before someone shuts it for you",
        "qrels": {"x_violent_speech": 2, "meta_bullying_section_4": 2, "x_abusive_behavior": 2}
    },
    {
        "query_id": "O25",
        "query": "publishing your GPS location every hour so strangers can follow you",
        "qrels": {"reddit_rule_3_privacy": 2, "meta_bullying_section_4": 2, "imda_osa_cyberbullying": 2}
    },

    # --- 25 Safe / Benign / Counter-speech Queries ---
    {
        "query_id": "S01",
        "query": "Can someone recommend a good beginner tutorial for Python and machine learning?",
        "qrels": {}
    },
    {
        "query_id": "S02",
        "query": "I am proud to be a gay man and no one can take my dignity away.",
        "qrels": {}
    },
    {
        "query_id": "S03",
        "query": "The weather in Singapore is unusually warm and humid this afternoon.",
        "qrels": {}
    },
    {
        "query_id": "S04",
        "query": "We must stand together and reject racism in all its ugly forms.",
        "qrels": {}
    },
    {
        "query_id": "S05",
        "query": "What is the best recipe for baking homemade chocolate chip cookies?",
        "qrels": {}
    },
    {
        "query_id": "S06",
        "query": "Our university library is open 24 hours during final exam week.",
        "qrels": {}
    },
    {
        "query_id": "S07",
        "query": "I strongly disagree with the prime minister's new economic fiscal policy.",
        "qrels": {}
    },
    {
        "query_id": "S08",
        "query": "Supporting immigrant families strengthens our community and local culture.",
        "qrels": {}
    },
    {
        "query_id": "S09",
        "query": "The referee made a terrible call in the football match yesterday.",
        "qrels": {}
    },
    {
        "query_id": "S10",
        "query": "Learning a new foreign language requires patience and daily practice.",
        "qrels": {}
    },
    {
        "query_id": "S11",
        "query": "Congratulations to all the graduating seniors on their huge milestone!",
        "qrels": {}
    },
    {
        "query_id": "S12",
        "query": "Please remember to recycle plastic bottles and aluminum cans.",
        "qrels": {}
    },
    {
        "query_id": "S13",
        "query": "Mental health awareness is critical for students facing high academic pressure.",
        "qrels": {}
    },
    {
        "query_id": "S14",
        "query": "Which laptop model has the longest battery life for data science programming?",
        "qrels": {}
    },
    {
        "query_id": "S15",
        "query": "I am hosting a community charity run this Saturday morning at the park.",
        "qrels": {}
    },
    {
        "query_id": "S16",
        "query": "The sunset over Marina Bay was breathtaking yesterday evening.",
        "qrels": {}
    },
    {
        "query_id": "S17",
        "query": "Islam teaches compassion, charity, and peace among neighbors.",
        "qrels": {}
    },
    {
        "query_id": "S18",
        "query": "Is it possible to fine-tune a BERT model on a single GPU with 16GB memory?",
        "qrels": {}
    },
    {
        "query_id": "S19",
        "query": "Women in STEM fields continue to inspire groundbreaking scientific discoveries.",
        "qrels": {}
    },
    {
        "query_id": "S20",
        "query": "Where can I buy fresh organic vegetables near Jurong West?",
        "qrels": {}
    },
    {
        "query_id": "S21",
        "query": "The subway trains were delayed this morning due to regular track maintenance.",
        "qrels": {}
    },
    {
        "query_id": "S22",
        "query": "Listening to classical instrumental music helps me focus while writing essays.",
        "qrels": {}
    },
    {
        "query_id": "S23",
        "query": "We adopted a rescue puppy from the animal shelter yesterday and we love him.",
        "qrels": {}
    },
    {
        "query_id": "S24",
        "query": "What are the key differences between transformer attention and convolution?",
        "qrels": {}
    },
    {
        "query_id": "S25",
        "query": "Wishing everyone a peaceful, healthy, and prosperous new year with family.",
        "qrels": {}
    }
]


def main():
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "qrels_policy.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(BENCHMARK_QRELS, f, indent=2)
    print(f"Successfully generated 75-query QRELS benchmark: {out_file} (25 Hate, 25 Harassment, 25 Safe)")


if __name__ == "__main__":
    main()
