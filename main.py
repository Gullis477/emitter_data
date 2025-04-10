from emitter_data import build_emitter,Emitter,EmitterConfig, plot_signal


if __name__ == "__main__":
    config = EmitterConfig.generate(PRI_MODULATION=3,FREQ_MODULATION=2)
    emitter = build_emitter(config)
    signal = emitter.signal()
    plot_signal(signal)
    emitter.signal_to_csv("path_to_csv")