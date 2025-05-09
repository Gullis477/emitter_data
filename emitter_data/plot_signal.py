import matplotlib.pyplot as plt


def plot_signal(df, save_path=None):
    if not {"freq", "pw", "bw", "toa", "pri"}.issubset(df.columns):
        raise ValueError(
            "DataFrame must contain 'freq', 'pw', 'bw', 'toa', and 'pri' columns"
        )

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Plot Frequency
    axs[0].scatter(df["toa"], df["freq"], color="tab:blue")
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].set_title("Frequency vs Time of Arrival")
    axs[0].grid(True)

    # Plot Pulse Width
    axs[1].scatter(df["toa"], df["pw"], color="tab:orange")
    axs[1].set_ylabel("Pulse Width (s)")
    axs[1].set_title("Pulse Width vs Time of Arrival")
    axs[1].grid(True)

    # Plot Bandwidth
    axs[2].scatter(df["toa"], df["bw"], color="tab:green")
    axs[2].set_ylabel("Bandwidth (Hz)")
    axs[2].set_title("Bandwidth vs Time of Arrival")
    axs[2].grid(True)

    # Plot PRI
    axs[3].scatter(df["toa"], df["pri"], color="red")
    axs[3].set_ylabel("PRI (s)")
    axs[3].set_xlabel("Time of Arrival (s)")
    axs[3].set_title("PRI vs Time of Arrival")
    axs[3].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    from .Emitter import build_emitter, EmitterConfig, Emitter

    pris = [1, 2, 3]
    freqs = [1, 2, 3]
    configs = []
    # for _ in range(10):
    config = EmitterConfig.generate(PRI_MODULATION=3, FREQ_MODULATION=1)
    configs.append(config)
    emitter = build_emitter(config=config)
    signal = emitter.signal(noise=True)
    print(signal.tail())

    df = signal
    plot_signal(df, save_path="signal_plot.png", static_axis=True)
    #    print(df[['freq', 'pw', 'bw', 'toa']].to_string(formatters={
    #        'pw': '{:.20f}'.format,
    #        'bw': '{:.20f}'.format,
    #    }))
