from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="t5-small")


# Input text (Example passage)
text = """
Professors Albus Dumbledore, Minerva McGonagall and gamekeeper Rubeus Hagrid from Hogwarts School of Witchcraft and Wizardry deliver the orphan, Harry Potter, to his only living relatives, the Dursleys. Harry grows up, unaware that he is a wizard and is led to believe his parents were killed in a car crash.

10 years later, owls begin delivering letters addressed to Harry. To prevent them, the Dursleys drag Harry to a deserted cabin where Hagrid arrives, confirming that Harry is a wizard and has been accepted to Hogwarts, having been lied to by the Dursleys. Hagrid brings Harry to Diagon Alley to purchase his school supplies, and buys him a snowy owl, which Harry names Hedwig. The core of Harry’s chosen wand has a feather from Dumbledore's phoenix, just like the wand of Lord Voldemort, the dark wizard. Hagrid tells Harry that Voldemort murdered his parents. But when he tried to kill Harry, his curse rebounded, leaving Harry with only a scar. Voldemort was defeated, for which Harry became famous.
"""

# Summarize the text
summary = summarizer(text, max_length=50, min_length=20, do_sample=False)

# Print the summary
print("Summary:", summary[0]['summary_text'])
