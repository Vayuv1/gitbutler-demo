import spacy

nlp = spacy.load('en_core_web_lg')

print("Calculate similarity")
templates = (
    "descend and maintain", 
    "ascend and maintain", 
    "maintain", 
    "turn left heading", 
    "turn right heading", 
    "heading"
)

d_temps = [nlp(temp) for temp in templates]

text = "descend n maintain"
d_text = nlp(text)

for d_temp in d_temps:
    print(d_text.similarity(d_temp))
