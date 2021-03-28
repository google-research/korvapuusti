package filter

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"math/cmplx"

	"github.com/google-research/korvapuusti/tools/synthesize/signals"
	"github.com/mjibson/go-dsp/fft"
)

type LTIConf struct {
	Gain  float64
	Poles []complex128
	Zeros []complex128
}

func (l LTIConf) Make() (*LTI, error) {
	if !l.Causal() {
		return nil, fmt.Errorf("anti-causal")
	}
	lti := &LTI{
		gain:       complex(l.Gain, 0),
		poleCoeffs: coeffs(l.Poles),
		zeroCoeffs: coeffs(l.Zeros),
		xHist:      make([]complex128, len(l.Poles)+1),
		yHist:      make([]complex128, len(l.Poles)+1),
	}
	return lti, nil
}

func (l LTIConf) Stable() bool {
	for _, pole := range l.Poles {
		if cmplx.Abs(pole) >= 1.0 {
			return false
		}
	}
	return true
}

func (l LTIConf) Causal() bool {
	return len(l.Poles) >= len(l.Zeros)
}

func (l LTIConf) Print(height, width int, w io.Writer) {
	wPerLine := math.Pi / float64(height)
	headers := []string{}
	gains := []float64{}
	maxHeaderLen := 0
	minGain := math.MaxFloat64
	maxGain := -math.MaxFloat64
	for i := 0; i < height; i++ {
		w := float64(i) * wPerLine
		header := fmt.Sprintf("%.2f ", w)
		if len(header) > maxHeaderLen {
			maxHeaderLen = len(header)
		}
		headers = append(headers, header)
		gain := cmplx.Abs(l.H(cmplx.Exp(complex(0, w))))
		if !math.IsInf(gain, 1) && gain > maxGain {
			maxGain = gain
		}
		if gain < minGain {
			minGain = gain
		}
		gains = append(gains, gain)
	}
	gainLen := width - maxHeaderLen
	widthPerGain := float64(gainLen) / (maxGain - minGain)
	for i := 0; i < height; i++ {
		header := bytes.NewBufferString(headers[i])
		for len(header.String()) < maxHeaderLen {
			fmt.Fprint(header, " ")
		}
		gainPart := &bytes.Buffer{}
		if math.IsInf(gains[i], 1) {
			for len(gainPart.String()) < gainLen {
				fmt.Fprintf(gainPart, "+Inf ")
			}
		} else {
			for len(gainPart.String()) < int((gains[i]-minGain)*widthPerGain) {
				fmt.Fprintf(gainPart, " ")
			}
			fmt.Fprintf(gainPart, "*")
		}
		fmt.Fprintf(w, "%v%v\n", header.String(), gainPart.String())
	}

}

func Hz2Z(f, rate signals.Hz) complex128 {
	return cmplx.Exp(complex(0, math.Pi*f/rate))
}

func (l LTIConf) Convolve(s []complex128) []complex128 {
	coeffs := fft.FFT(s)
	wPerBin := 2 * math.Pi / float64(len(coeffs))
	for i := range coeffs {
		z := cmplx.Exp(complex(0, wPerBin*float64(i)))
		coeffs[i] *= l.H(z)
	}
	return fft.IFFT(coeffs)
}

func (l LTIConf) H(z complex128) complex128 {
	res := complex(l.Gain, 0)
	for _, zero := range l.Zeros {
		res *= z - zero
	}
	denom := complex128(1)
	for _, pole := range l.Poles {
		denom *= z - pole
	}
	return res / denom
}

func (l LTIConf) HEquation(numbers bool) string {
	res := &bytes.Buffer{}
	if numbers {
		fmt.Fprintf(res, "%v * ", l.Gain)
	} else {
		fmt.Fprintf(res, "g * ")
	}
	for i := range l.Zeros {
		if numbers {
			fmt.Fprintf(res, "(z - %v)", l.Zeros[i])
		} else {
			fmt.Fprintf(res, "(z - q%v)", i)
		}
		if i+1 < len(l.Zeros) {
			fmt.Fprintf(res, " * ")
		}
	}
	fmt.Fprintf(res, " / ")
	for i := range l.Poles {
		if numbers {
			fmt.Fprintf(res, "(z - %v)", l.Poles[i])
		} else {
			fmt.Fprintf(res, "(z - p%v)", i)
		}
		if i+1 < len(l.Poles) {
			fmt.Fprintf(res, " * ")
		}
	}
	fmt.Fprintf(res, ")")
	return res.String()
}

type LTI struct {
	gain       complex128
	poleCoeffs []complex128
	zeroCoeffs []complex128
	xHist      []complex128
	yHist      []complex128
	histIdx    int
}

func MakePZ(params [][2]float64) []complex128 {
	res := []complex128{}
	for _, pair := range params {
		z := complex(pair[0], 0) * cmplx.Exp(complex(0, pair[1]))
		res = append(res, z)
		res = append(res, cmplx.Conj(z))
	}
	return res
}

func (l *LTI) Preload(x, y complex128) {
	l.setX(x)
	l.setY(y)
	l.incHist()
}

func (l *LTI) Y(x complex128) complex128 {
	l.setX(x)
	res := complex128(0)
	zeroCoeffExponentOffset := 0
	if len(l.poleCoeffs) > len(l.zeroCoeffs) {
		zeroCoeffExponentOffset = len(l.poleCoeffs) - len(l.zeroCoeffs)
	}
	for i := range l.zeroCoeffs {
		//fmt.Printf("res += x[n-%v] * g * zc%v => %v += %v * %v * %v = %v\n", res, i+zeroCoeffExponentOffset, i, l.getX(-(i + zeroCoeffExponentOffset)), l.gain, l.zeroCoeffs[i], res+l.getX(-(i+zeroCoeffExponentOffset))*l.gain*l.zeroCoeffs[i])
		res += l.getX(-(i + zeroCoeffExponentOffset)) * l.gain * l.zeroCoeffs[i]
	}
	for i := 1; i < len(l.poleCoeffs); i++ {
		//fmt.Printf("res -= y[n-%v] * pc%v => %v += %v * %v = %v\n", i, i, res, l.getY(-i), l.poleCoeffs[i], res-l.getY(-i)*l.poleCoeffs[i])
		res -= l.getY(-i) * l.poleCoeffs[i]
	}
	res /= l.poleCoeffs[0]
	l.setY(res)
	l.incHist()
	//fmt.Printf("returning %v\n", res)
	return res
}

// H = Y/X = g * (z - q1) * (z - q2) * ... (z - qn) / ( (z - p1) * (z - p2) * ... (z - pn) )
// Y/X = g * (1 - q1 / z) * (1 - q2 / z) * ... (1 - qn / z) / ( (1 - p1 / z) * (1 - p2 / z) * ... (1 - pn / z) )
// Y/X = g * (qc0 + qc1 / z + qc2 / z^2 + ... qcn / z^n) / (pc0 + pc1 / z + pc2 / z^2 + ... pcn / z^n)
// Y * (pc0 + pc1 / z + pc2 / z^2 + ... pcn / z^n) = X * g * (qc0 + qc1 / z + qc2 / z^2 + ... qcn / z^n)
// Y * pc0 + Y * pc1 / z + Y * pc2 / z^w + ... Y * pcn / z^n = X * g * qc0 + X * g * qc1 / z + X * g * qc2 / z^2 + ... X * g * qcn / z^n
// Y * pc0 = X * g * qc0 + X * g * qc1 / z + X * g * qc2 / z^2 + ... X * g * qcn / z^n - (Y * pc1 / z + Y * pc2 / z^w + ... Y * pcn / z^n)
// Y = (X * g * qc0 + X * g * qc1 / z + X * g * qc2 / z^2 + ... X * g * qcn / z^n - (Y * pc1 / z + Y * pc2 / z^2 + ... Y * pcn / z^n)) / pc0
// y[t] = (x[t] * g * qc0 + x[t-1] * g * qc1 + x[t-2] * g * qc2 + ... x[t-n] * g * qcn - (y[t-1] * pc1 + y[t-2] * pc2 + ... y[t-n] * pcn)) / pc0
func (l *LTI) DifferenceEquation(numbers bool) string {
	res := &bytes.Buffer{}
	fmt.Fprintf(res, "(")
	zeroCoeffExponentOffset := 0
	if len(l.poleCoeffs) > len(l.zeroCoeffs) {
		zeroCoeffExponentOffset = len(l.poleCoeffs) - len(l.zeroCoeffs)
	}
	for i := range l.zeroCoeffs {
		step := fmt.Sprintf("n-%v", i+zeroCoeffExponentOffset)
		if i+zeroCoeffExponentOffset == 0 {
			step = "n"
		}
		if numbers {
			if l.gain*l.zeroCoeffs[i] == 0 {
				fmt.Fprint(res, "0")
			} else {
				if l.gain*l.zeroCoeffs[i] == 1 {
					fmt.Fprintf(res, "x[%v]", step)
				} else {
					fmt.Fprintf(res, "x[%v] * %v", step, l.gain*l.zeroCoeffs[i])
				}
			}
		} else {
			fmt.Fprintf(res, "x[%v] * g * qc%v", step, i)
		}
		if i+1 < len(l.zeroCoeffs) {
			fmt.Fprintf(res, " + ")
		}
	}
	fmt.Fprintf(res, " - (")
	for i := 1; i < len(l.poleCoeffs); i++ {
		if numbers {
			if l.poleCoeffs[i] == 0 {
				fmt.Fprint(res, "0")
			} else {
				if l.poleCoeffs[i] == 1 {
					fmt.Fprintf(res, "y[n-%v]", i)
				} else {
					fmt.Fprintf(res, "y[n-%v] * %v", i, l.poleCoeffs[i])
				}
			}
		} else {
			fmt.Fprintf(res, "y[n-%v] * pc%v", i, i)
		}
		if i+1 < len(l.poleCoeffs) {
			fmt.Fprintf(res, " + ")
		}
	}
	if numbers {
		if l.poleCoeffs[0] == 1 {
			fmt.Fprintf(res, "))")
		} else {
			fmt.Fprintf(res, ")) / %v", l.poleCoeffs[0])
		}
	} else {
		fmt.Fprintf(res, ")) / pc0")
	}
	return res.String()
}

func (l *LTI) getX(d int) complex128 {
	return l.xHist[(len(l.xHist)+l.histIdx+d)%len(l.xHist)]
}

func (l *LTI) setX(v complex128) {
	if len(l.xHist) == 0 {
		return
	}
	l.xHist[l.histIdx] = v
}

func (l *LTI) incHist() {
	l.histIdx = (l.histIdx + 1) % len(l.xHist)
}

func (l *LTI) getY(d int) complex128 {
	return l.yHist[(len(l.yHist)+l.histIdx+d)%len(l.yHist)]
}

func (l *LTI) setY(v complex128) {
	if len(l.yHist) == 0 {
		return
	}
	l.yHist[l.histIdx] = v
}

// takeNumOfLength returns all combinations of num elements from a set of length.
func takeNumOfLength(length, num int) [][]int {
	var helper func(int, int) [][]int
	helper = func(pos, remaining int) [][]int {
		combos := [][]int{}
		for i := pos; i < length; i++ {
			if remaining == 1 {
				combos = append(combos, []int{i})
			} else {
				for _, remainder := range helper(i+1, remaining-1) {
					combo := []int{i}
					combo = append(combo, remainder...)
					combos = append(combos, combo)
				}
			}
		}
		return combos
	}
	return helper(0, num)
}

// coeffs creates coefficients matching the following pattern:
// (1 - k1 * x) * (1 - k2 * x) * ... (1 - kn * x) = c0 + c1 * x + c2 * x^2 + ... cn * x^n
func coeffs(constants []complex128) []complex128 {
	res := make([]complex128, len(constants)+1)
	for num := 0; num <= len(constants); num++ {
		sum := complex128(0)
		if num == 0 {
			sum = 1
		}
		combos := takeNumOfLength(len(constants), num)
		for _, parts := range combos {
			prod := complex128(1)
			for _, part := range parts {
				prod *= -constants[part]
			}
			sum += prod
		}
		res[num] = sum
	}
	return res
}
