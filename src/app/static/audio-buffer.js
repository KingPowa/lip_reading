class AudioBufferQueue {
  constructor(settings) {
    this.settings = settings;
    this.chunks = [];
    this.latestEnd = 0;
    this.acknowledgedUntil = 0;
    this.lastSubmittedEnd = 0;
    this.initialized = false;
  }

  append(samples, offset) {
    if (!samples.length) return;
    if (
      this.initialized &&
      Math.abs(offset - this.latestEnd) > 1 / this.settings.sampleRate
    )
      throw new Error("Microphone audio has a discontinuity. Restart capture.");
    if (!this.initialized) {
      this.acknowledgedUntil = offset;
      this.lastSubmittedEnd = offset;
      this.initialized = true;
    }
    const end = offset + samples.length / this.settings.sampleRate;
    this.chunks.push({ samples, offset, end });
    this.latestEnd = end;
  }

  acknowledge(timestamp) {
    this.acknowledgedUntil = Math.max(this.acknowledgedUntil, timestamp);
    const oldestNeeded = this.acknowledgedUntil;
    while (this.chunks.length && this.chunks[0].end <= oldestNeeded)
      this.chunks.shift();
  }

  snapshot(stopping) {
    if (!this.chunks.length) {
      if (!stopping || !this.initialized) return undefined;
      return {
        samples: new Int16Array(0),
        offset: this.latestEnd,
        end: this.latestEnd,
        final: true,
      };
    }
    const origin = this.chunks[0].offset;
    const rate = this.settings.sampleRate;
    const startSample = Math.max(
      0,
      Math.round((this.acknowledgedUntil - origin) * rate),
    );
    const offset = origin + startSample / rate;
    const length = Math.min(
      Math.round((this.latestEnd - offset) * rate),
      Math.round(this.settings.maximumWindowSeconds * rate),
    );
    if (length <= 0) return undefined;
    const end = offset + length / rate;
    const samples = new Int16Array(length);
    for (const chunk of this.chunks) {
      const sourceStart = Math.max(
        0,
        Math.round((offset - chunk.offset) * rate),
      );
      const sourceEnd = Math.min(
        chunk.samples.length,
        Math.round((end - chunk.offset) * rate),
      );
      if (sourceEnd <= sourceStart) continue;
      const destinationStart = Math.max(
        0,
        Math.round((chunk.offset - offset) * rate),
      );
      samples.set(
        chunk.samples.subarray(sourceStart, sourceEnd),
        destinationStart,
      );
    }
    return {
      samples,
      offset,
      end,
      final: stopping && end >= this.latestEnd - 1 / rate,
    };
  }

  get backlogSeconds() {
    return Math.max(0, this.latestEnd - this.acknowledgedUntil);
  }
}
