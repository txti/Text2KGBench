import time

from kgbench.utils.llm import get_llm_response

# MODEL = "qwen2.5:7b-instruct"
MODEL = "qwen2.5-coder:1.5b-instruct"

sentences = ["A Gang Story (French: Les Lyonnais) is a 2011 French drama film directed by Olivier Marchal.",
            "Released in Italy in 1964 and then in the United States in 1967, A Fistful of Dollars initiated the popularity of the Spaghetti Western genre.",
            "The Dark Knight Rises is a 2012 superhero film directed by Christopher Nolan, who co-wrote the screenplay with his brother Jonathan Nolan, and the story with David S. Goyer."]

sentence_context = ["Visions of Light is a 1992 documentary film directed by Arnold Glassman, Todd McCarthy and Stuart Samuels.",
                    "Visions of Light film genre is Science and Documentary",
                    "The Visions of Light film screenplay was done by Quentin Tarantino and John Travolta"]

correct_answers = ["director(Visions of Light, Arnold Glassman) \ndirector(Visions of Light, Todd McCarthy) \ndirector(Visions of Light, Stuart Samuels)",
                    "genre(Visions of Light, Science)\ngenre(Visions of Light, Documentary)",
                    "screenplay(Visions of Light, Quentin Tarantino)\nscreenplay(Visions of Light, John Travolta)"]

# standard prompt
prompt="""
Given the following ontology and sentence, extract the triples from the sentence according to the relations in the ontology in the following Json format:

[
    {
        "subject": "subject_name",
        "predicate": "film|film genre|film production company|film award|city|human",
        "object": "object_name"
    }
]

Only include the triples in the provided format with no additional text.

Context:
Ontology Concepts: film, film genre, film production company, film award, city, human
Ontology relations: director(film, human), genre(film, film genre), cast member(film, human)

Sentence:
"""

correct_answer_text="\nCorrect Answer:\n"
sent_text = "\nGiven this sentence: "

answer="""
Give the correct answer
Answer:
"""

batch_start = time.time()

for i in range(len(sentences)):
    prompt_sentence = prompt + sentence_context[i]

    prompt_sentence += sent_text + sentences[i] + answer
    print(f"\nREQUEST:\n{'-'*80}\n{prompt_sentence}")

    # Prompt sent to the LLM
    start_time = time.time()
    response = get_llm_response(prompt_sentence, model=MODEL)
    end_time = time.time()
    print(f"\nRESPONSE: ({end_time - start_time} ms)\n{'-'*80}\n{response}\n")

    # prompt_sentence += correct_answer_text + correct_answers[i]

batch_end = time.time()
print(f"\nBatch total time: {batch_end-batch_start}")
