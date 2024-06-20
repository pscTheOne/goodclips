<b>The idea is to extract audio from a film or tv episode and:</b>

1. use a vad to isolate and cut out sentences: <b>vad-it.py</b>
2. run wisper on the clips to transcibe them: <b>transcribe-it.py</b>
3. Create a vector for each text sentence to compare meaning: <b>sentence-it.py</b>
4. query chatgpt with a promt, take the result and vectorize it and return the closest matching sentence and play the coresponding wav file: <b>promt.py</b>
