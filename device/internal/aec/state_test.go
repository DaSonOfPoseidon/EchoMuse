package aec

import (
	"encoding/binary"
	"math"
	"testing"
)

// hwCanceller is a canceller on the hardware-reference path, as the device
// runs it.
func hwCanceller(tailMs int) *Canceller {
	c := New()
	c.SetParams(true, 0, tailMs)
	c.SetHardwareRef(true)
	return c
}

// firstFramesAttenuation runs the echo scenario from
// TestHardwareRefCancelsWithoutRingOrDelay (33 samples, inverted) and
// returns the cancellation over the first n frames, in dB.
func firstFramesAttenuation(c *Canceller, signal []int16, n int) float64 {
	var in, out float64
	for f := 0; f < n; f++ {
		lo, hi := f*FrameSize, (f+1)*FrameSize
		mic := make([]int16, FrameSize)
		for i := range mic {
			if j := lo + i - 33; j >= 0 {
				mic[i] = -signal[j]
			}
		}
		mb := toBytes(mic)
		res := c.ProcessWithRef(mb, toBytes(signal[lo:hi]))
		in += rms(mb)
		out += rms(res)
	}
	return 20 * math.Log10(in/out)
}

// The point of saving the echo path: a canceller loaded with one cancels
// from its first frames, where a cold one is still learning. Measured on
// the bench as 0-1dB for the first seconds of a reply from cold (#aec
// harness, 2026-09-22).
func TestLoadedStateCancelsFromTheFirstFrame(t *testing.T) {
	signal := synth(200 * FrameSize)
	warm := hwCanceller(64)
	firstFramesAttenuation(warm, signal, 200)
	saved, err := warm.ExportState()
	if err != nil {
		t.Fatal(err)
	}

	other := synth(20*FrameSize + 7) // different audio from the one learnt on
	other = other[7:]
	cold := firstFramesAttenuation(hwCanceller(64), other, 10)
	loaded := hwCanceller(64)
	if err := loaded.ImportState(saved); err != nil {
		t.Fatal(err)
	}
	warmStart := firstFramesAttenuation(loaded, other, 10)
	t.Logf("first 10 frames: cold %.1fdB, loaded %.1fdB", cold, warmStart)
	if warmStart < cold+10 {
		t.Fatalf("loaded state did not help: cold %.1fdB, loaded %.1fdB", cold, warmStart)
	}
}

func TestImportRefusesAMismatchedFilter(t *testing.T) {
	src := hwCanceller(64)
	saved, _ := src.ExportState()
	if err := hwCanceller(128).ImportState(saved); err == nil {
		t.Fatal("a state saved at 64ms must not load into a 128ms filter")
	}
}

func TestImportRefusesNonFiniteValues(t *testing.T) {
	c := hwCanceller(64)
	saved, _ := c.ExportState()
	binary.LittleEndian.PutUint32(saved[len(saved)-4:], math.Float32bits(float32(math.NaN())))
	if err := hwCanceller(64).ImportState(saved); err == nil {
		t.Fatal("a NaN in the saved filter must be refused")
	}
}

func TestImportRefusesGarbage(t *testing.T) {
	c := hwCanceller(64)
	for _, b := range [][]byte{nil, []byte("EMAEC1"), []byte("not a state at all, clearly")} {
		if err := c.ImportState(b); err == nil {
			t.Fatalf("accepted %q", b)
		}
	}
}

func TestExportNeedsARunningCanceller(t *testing.T) {
	if _, err := New().ExportState(); err == nil {
		t.Fatal("export from a disabled canceller must fail")
	}
}
