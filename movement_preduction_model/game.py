# Ontwikkeld door Ahmad Al Dibo
# Ontwikkelde datum: 2025/08/09 augustus
# Versie: 1.0


from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import pickle
import numpy as np
import pygame


sen_length = 100
directions = ["boven", "beneden", "links", "rechts"]
directions2Engels = {"boven": "up", "beneden": "down", "links": "left", "rechts": "right"}
word2int = {"up": 0, "down": 1, "left": 2, "right": 3}
int2word = {v: k for k, v in word2int.items()}


tokenizer_path = "data\\tokenizer_mulit_output_model.pkl"
model_path = "models\\multi_output_model.h5"

with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

def prediction(model, sms, tokenizer):
    """
    Predict the direction based on the input SMS.
    This is for one output (direction).
    you can use lstm_movement_model_preduction.h5 and tokenizer_one_output_model.pkl

    """
    seq = tokenizer.texts_to_sequences([sms])
    seq = pad_sequences(seq, maxlen=sen_length)
    output = model.predict(seq)
    predicted_index = np.argmax(output, axis=1)[0]
    predicted_word = directions[predicted_index]
    predicted_word = directions2Engels[predicted_word] 
    return predicted_word

model=load_model(model_path)

def prediction_multi(model, sms, tokenizer):
    seq = tokenizer.texts_to_sequences([sms])
    seq = pad_sequences(seq, maxlen=sen_length)
    pred_dir, pred_steps = model.predict(seq)
    pred_dir_index = np.argmax(pred_dir, axis=1)[0]
    pred_steps_index = np.argmax(pred_steps, axis=1)[0]

    direction = int2word[pred_dir_index]
    steps = pred_steps_index + 1 

    return direction, steps


DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0)
}

def move(pos, direction, steps):
    dx, dy = DIRECTIONS[direction]
    return [pos[0] + dx * steps, pos[1] + dy * steps]

pygame.init()
screen_size = 500
tile_size = 50
cols = rows = screen_size // tile_size

screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("AI Beweeg Spel")

clock = pygame.time.Clock()
player_pos = [5, 5]

running = True
commands = []

while running:
    screen.fill((30, 30, 30))

    for x in range(0, screen_size, tile_size):
        for y in range(0, screen_size, tile_size):
            rect = pygame.Rect(x, y, tile_size, tile_size)
            pygame.draw.rect(screen, (50, 50, 50), rect, 1)

    px, py = player_pos
    pygame.draw.rect(screen, (0, 255, 0), (px * tile_size, py * tile_size, tile_size, tile_size))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                user_command = input("Typ een opdracht (zoals 'ga twee stappen naar boven '): ")
                direction, steps = prediction_multi(model, user_command, tokenizer)

                commands = [{"direction": direction, "steps": steps}]

                for cmd in commands:
                    direction = cmd.get("direction")
                    steps = cmd.get("steps", 1)
                    if direction in DIRECTIONS:
                        player_pos = move(player_pos, direction, steps)
                        player_pos[0] = max(0, min(cols - 1, player_pos[0]))
                        player_pos[1] = max(0, min(rows - 1, player_pos[1]))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()