class MicrophoneProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const settings = options.processorOptions;
    this.outputRate = settings.sampleRate;
    this.packetSamples = Math.round(settings.packetSeconds * this.outputRate);
    this.samples = new Int16Array(this.packetSamples);
    this.usedSamples = 0;
    this.totalSamples = 0;
    this.inputPerOutput = sampleRate / this.outputRate;
    this.accumulatedWeight = 0;
    this.accumulatedValue = 0;
    this.originTime = undefined;
    this.stopped = false;
    this.port.onmessage = (event) => {
      if (event.data !== "flush") return;
      this.sendPacket();
      this.stopped = true;
      this.port.postMessage({ stopped: true });
    };
  }

  sendPacket() {
    if (!this.usedSamples) return;
    const samples = this.samples.slice(0, this.usedSamples);
    const timestamp = this.originTime + this.totalSamples / this.outputRate;
    this.totalSamples += this.usedSamples;
    this.usedSamples = 0;
    this.port.postMessage({ samples, timestamp }, [samples.buffer]);
  }

  process(inputs) {
    if (this.stopped) return false;
    const channel = inputs[0]?.[0];
    if (!channel) return true;
    this.originTime ??= currentTime;
    // Integrate input samples across output intervals; retain the resampling phase between render quanta.
    for (const sample of channel) {
      let remainingWeight = 1;
      while (remainingWeight > 0) {
        const consumedWeight = Math.min(
          remainingWeight,
          this.inputPerOutput - this.accumulatedWeight,
        );
        this.accumulatedValue += sample * consumedWeight;
        this.accumulatedWeight += consumedWeight;
        remainingWeight -= consumedWeight;
        if (this.accumulatedWeight + Number.EPSILON < this.inputPerOutput)
          continue;
        const value = Math.max(
          -1,
          Math.min(1, this.accumulatedValue / this.inputPerOutput),
        );
        this.samples[this.usedSamples++] = Math.round(value * 32767);
        this.accumulatedValue = 0;
        this.accumulatedWeight = 0;
        if (this.usedSamples === this.packetSamples) this.sendPacket();
      }
    }
    return true;
  }
}

registerProcessor("microphone-pcm", MicrophoneProcessor);
