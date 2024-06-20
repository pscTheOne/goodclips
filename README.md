<h2>The idea is to extract audio from a film or tv episode and:</h2>

<h3>Installation</h3>

```

python -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt

```

<h3>Usage</h3>

1. use a vad to isolate and cut out sentences: <b>vad-it.py <agressiveness> <video file></b> <i>agressiveness is a int, either 1, 2 or 3. experiment, I have had success with 3.</i>
2. run wisper on the clips to transcibe them: <b>transcribe-it.py</b>
3. Create a vector for each text sentence to compare meaning: <b>sentence-it.py</b>
4. query chatgpt with a promt, take the result and vectorize it and return the closest matching sentence and play the coresponding wav file: <b>promt.py</b>
