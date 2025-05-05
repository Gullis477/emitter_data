run 

```
pip install -r requirements.txt
```

example
```py
from emitter_data import build_emitter,Emitter,EmitterConfig, plot_signal


if __name__ == "__main__":
    config = EmitterConfig.generate(PRI_MODULATION=3,FREQ_MODULATION=2)
    emitter = build_emitter(config)
    signal = emitter.signal(noise = False)
    plot_signal(signal)

```

supported pri and frequency types:
```py
    PRI_MODULATION: Literal[1, 2, 3]  # ["jitter", "stagger", "dwell_and_switch"]
    FREQ_MODULATION: Literal[1, 2, 3] # ["static","jitter","pri dependent"]
```
