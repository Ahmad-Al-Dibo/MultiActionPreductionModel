# MultiActionPreductionModel
Mulit Action model voor 2D game om blockje te verplatsen met input text

# Movement Command Classification Model

Dit project bevat een LSTM-neuraal netwerkmodel dat getraind is om Nederlandse commando’s te classificeren die een beweging beschrijven. Het model herkent twee dingen tegelijk uit een zin:

- **De richting**: boven, beneden, links, rechts
- **Het aantal stappen**: één tot vijf

---

## Dataset

Afmetingen(embeding_matrix(glove.6B.{nummer})): [GloVe embeddings - Stanford NLP](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbXo0Z2RibXgyZm1oWG1YV25DZE9xWEc0R3lZZ3xBQ3Jtc0tuVkVvZENLUUQyd0xsWS1tZUxpZlhMU2J3ekdiWFJSd0UwdVFtNmthOXV6MUlaV0FTR1JxUlhlWjloUDRwbEx2ZXVIajIwY3RUYjJfT3BMZ0ZtVldhcERQLWR3SWl3QmVIUWh0M1NQOEtIZVdTWmJkcw&q=https%3A%2F%2Fnlp.stanford.edu%2Fprojects%2Fglove%2F&v=HKcvvooqYpM)  jij hebt glove.6B.zip nodig [Klik hier om meteen te downloaden](https://nlp.stanford.edu/data/glove.6B.zip)


De dataset is synthetisch gegenereerd met zinnen in de vorm:


Voorbeelden:
- "ga twee stappen naar beneden"
- "loop drie stappen naar links"
- "beweeg één stappen naar rechts"

De richtingen en acties zijn vertaald naar Engelse labels voor het model.

---

## Model Architectuur

- Embedding layer met voorgetrainde GloVe-woordenvectoren (100d)
- LSTM laag met 128 units
- Twee output lagen:
  - Softmax classificatie over 4 richtingen
  - Softmax classificatie over 5 stap-aantallen

Het model is gebouwd met TensorFlow/Keras en gebruikt `categorical_crossentropy` als loss voor beide outputs.

---

## Training

- Dataset van 15.000 voorbeelden
- Train/test split 70% / 30%
- Batch size 64, 10 epochs
- Validatieaccuracy wordt gelogd voor beide outputs

<img width="1019" height="470" alt="output" src="https://github.com/user-attachments/assets/37eaf15e-3e34-4fc3-a8d4-9043656357e9" />

---

## Gebruik

Voorbeeld om een zin te voorspellen:

```python
direction, steps = prediction_multi(model, "ga twee stappen naar beneden", tokenizer)
print(f"Richting: {direction}, Stappen: {steps}")
```

# Auteur
## Ahmad Mahmoud Al Dibo
