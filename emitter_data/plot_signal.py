
import matplotlib.pyplot as plt

def plot_signal(df):
    if not {'freq', 'pw', 'bw', 'toa'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'freq', 'pw', 'bw', and 'toa' columns")

    plt.figure(figsize=(12, 8))

    # Plot freq
    plt.subplot(4, 1, 1)
    plt.scatter(df['toa'], df['freq'], label='Frequency', color='tab:blue')
    plt.ylabel('Frequency (Hz)')
    plt.title('Frequency vs Time of Arrival')
    plt.grid(True)

    # Plot pw
    plt.subplot(4, 1, 2)
    plt.scatter(df['toa'], df['pw'], label='Pulse Width', color='tab:orange')
    plt.ylabel('Pulse Width (s)')
    plt.title('Pulse Width vs Time of Arrival')
    plt.grid(True)

    # Plot bw
    plt.subplot(4, 1, 3)
    plt.scatter(df['toa'], df['bw'], label='Bandwidth', color='tab:green')
    plt.ylabel('Bandwidth (Hz)')
    plt.xlabel('Time of Arrival (s)')
    plt.title('Bandwidth vs Time of Arrival')
    plt.grid(True)


    # Plot bw
    plt.subplot(4, 1, 4)
    plt.scatter(df['toa'], df['pri'], label='PRI', color='red')
    plt.ylabel('PRI (s)')
    plt.xlabel('Time of Arrival (s)')
    plt.title('PRI vs Time of Arrival')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from .Emitter import build_emitter,EmitterConfig,Emitter
    pris = [1, 2, 3]
    freqs = [1, 2, 3]
    configs = []
    #for _ in range(10):
    config = EmitterConfig.generate(PRI_MODULATION=3,FREQ_MODULATION=1)
    configs.append(config)
    emitter = build_emitter(config=config)
    signal=emitter.signal(noise = True)
    print(signal.tail())

    df = signal
    plot_signal(df)
    #    print(df[['freq', 'pw', 'bw', 'toa']].to_string(formatters={
    #        'pw': '{:.20f}'.format,
    #        'bw': '{:.20f}'.format,
    #    }))