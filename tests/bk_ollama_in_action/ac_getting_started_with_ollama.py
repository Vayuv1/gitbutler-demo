import ollama


response = ollama.list()
print(response)

model = "gemma3"

# == Chat example ==
res = ollama.chat(
    model=model,
    messages=[
        {"role": "user", "content": "Say hello to user."},
    ],
)
print(res["message"]["content"])

# # == Chat example streaming ==
# res = ollama.chat(
#     model=model,
#     messages=[
#         {
#             "role": "user",
#             "content": "why is the ocean so salty?",
#         },
#     ],
#     stream=True,
# )
# for chunk in res:
#     print(chunk["message"]["content"], end="", flush=True)


# # ==================================================================================
# # ==== The Ollama Python library's API is designed around the Ollama REST API ====
# # ==================================================================================

# # == Generate example ==
# res = ollama.generate(
#     model=model,
#     prompt="why is the sky blue?",
# )

# print(ollama.show(model))


# Create a new model with modelfile
# modelfile = """
# FROM gemma3
# SYSTEM You are very smart assistant who knows everything about oceans. You are very succinct and informative.
# PARAMETER temperature 0.1
# """

# ollama.create(model="knowitall", modelfile=modelfile)

# res = ollama.generate(model="knowitall", prompt="why is the ocean so salty?")
# print(res["response"])


# # delete model
# ollama.delete("knowitall")
