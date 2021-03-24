package filter

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"sort"
	"testing"

	"github.com/google-research/korvapuusti/tools/synthesize/signals"
	"gonum.org/v1/gonum/stat"
)

type combinationSlice [][]int

func (c combinationSlice) Len() int {
	return len(c)
}

func (c combinationSlice) Less(i, j int) bool {
	sort.Sort(sort.IntSlice(c[i]))
	sort.Sort(sort.IntSlice(c[j]))
	return c[i][0] < c[j][0]
}

func (c combinationSlice) Swap(i, j int) {
	c[i], c[j] = c[j], c[i]
}

func checkEq(a, b [][]int) error {
	sort.Sort(combinationSlice(a))
	sort.Sort(combinationSlice(b))
	if len(a) != len(b) {
		return fmt.Errorf("lengths: %v != %v", len(a), len(b))
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return fmt.Errorf("length of elements %+v/%+v: %v != %v", a[i], b[i], len(a[i]), len(b[i]))
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return fmt.Errorf("elements %+v/%+v: %v != %v", a[i], b[i], a[i][j], b[i][j])
			}
		}
	}
	return nil
}

func TestMemory(t *testing.T) {
	lti := LTIConf{
		Poles: []complex128{0, 0, 0, 0},
		Zeros: []complex128{0, 0, 0},
	}.Make()
	for _, tc := range []struct {
		xBefore []complex128
		xPush   complex128
		yBefore []complex128
		yPush   complex128
	}{
		{
			xBefore: []complex128{0, 0, 0, 0},
			xPush:   complex(1, 0),
			yBefore: []complex128{0, 0, 0},
			yPush:   complex(0, 1),
		},
		{
			xBefore: []complex128{complex(1, 0), 0, 0, 0},
			xPush:   complex(2, 0),
			yBefore: []complex128{complex(0, 1), 0, 0},
			yPush:   complex(0, 2),
		},
		{
			xBefore: []complex128{complex(2, 0), complex(1, 0), 0, 0},
			xPush:   complex(3, 0),
			yBefore: []complex128{complex(0, 2), complex(0, 1), 0},
			yPush:   complex(0, 3),
		},
		{
			xBefore: []complex128{complex(3, 0), complex(2, 0), complex(1, 0), 0},
			xPush:   complex(4, 0),
			yBefore: []complex128{complex(0, 3), complex(0, 2), complex(0, 1)},
			yPush:   complex(0, 4),
		},
		{
			xBefore: []complex128{complex(4, 0), complex(3, 0), complex(2, 0), complex(1, 0)},
			xPush:   complex(5, 0),
			yBefore: []complex128{complex(0, 4), complex(0, 3), complex(0, 2)},
			yPush:   complex(0, 5),
		},
		{
			xBefore: []complex128{complex(5, 0), complex(4, 0), complex(3, 0), complex(2, 0)},
			xPush:   complex(6, 0),
			yBefore: []complex128{complex(0, 5), complex(0, 4), complex(0, 3)},
			yPush:   complex(0, 6),
		},
	} {
		for i, want := range tc.xBefore {
			if got := lti.x(-i); got != want {
				t.Errorf("got x[-%v] %v, wanted %v", i, got, want)
			}
		}
		lti.pushX(tc.xPush)
		for i, want := range tc.yBefore {
			if got := lti.y(-i); got != want {
				t.Errorf("got y[-%v] %v, wanted %v", i, got, want)
			}
		}
		lti.pushY(tc.yPush)
	}

}

func TestImpulseResponse(t *testing.T) {
	for _, tc := range []struct {
		conf LTIConf
	}{
		{
			conf: LTIConf{
				Gain:  0.5,
				Zeros: []complex128{-1},
				Poles: []complex128{1},
			},
		},
		/*
			{
				conf: LTIConf{
					Gain:  1,
					Zeros: MakePZ([][2]float64{{0.9, 3 * math.Pi / 4}, {0.7, math.Pi / 4}}),
					Poles: MakePZ([][2]float64{{0.8, math.Pi / 2}}),
				},
			},
		*/
	} {
		lti := tc.conf.Make()
		fmt.Printf("%+v\n", lti)
		s := make(signals.Float64Slice, 1000)
		s[0] = real(lti.Next(complex(1, 0)))
		for i := 1; i < len(s); i++ {
			s[i] = real(lti.Next(complex(0, 0)))
		}
		fmt.Printf("%+v\n", s.SpectrumGains())
		tc.conf.Print(100, 140, os.Stdout)
	}
}

func XXXTestFilter(t *testing.T) {
	for i, tc := range []struct {
		conf      LTIConf
		z         complex128
		wantedAmp float64
	}{
		{
			conf: LTIConf{
				Gain:  0.5,
				Zeros: []complex128{-1},
				Poles: []complex128{1},
			},
			z:         cmplx.Exp(complex(0, 0.1)),
			wantedAmp: 9.99,
		},
		{
			conf: LTIConf{
				Gain:  0.5,
				Zeros: []complex128{-1},
				Poles: []complex128{1},
			},
			z:         cmplx.Exp(complex(0, 0.5)),
			wantedAmp: 1.94,
		},
		{
			conf: LTIConf{
				Gain:  1,
				Zeros: []complex128{0.8 * cmplx.Exp(complex(0, 3*math.Pi/4)), cmplx.Conj(0.8 * cmplx.Exp(complex(0, 3*math.Pi/4)))},
				Poles: []complex128{0.8 * cmplx.Exp(complex(0, math.Pi/4)), cmplx.Conj(0.8 * cmplx.Exp(complex(0, math.Pi/4)))},
			},
			z:         cmplx.Exp(complex(0, 1)),
			wantedAmp: 5.2,
		},
		{
			conf: LTIConf{
				Gain:  1,
				Zeros: MakePZ([][2]float64{{0.9, 3 * math.Pi / 4}, {0.7, math.Pi / 4}}),
				Poles: MakePZ([][2]float64{{0.8, math.Pi / 2}}),
			},
			z:         cmplx.Exp(complex(0, 1)),
			wantedAmp: 1.15,
		},
		{
			conf: LTIConf{
				Gain:  1,
				Zeros: MakePZ([][2]float64{{0.9, 3 * math.Pi / 4}, {0.7, math.Pi / 4}}),
				Poles: MakePZ([][2]float64{{0.8, math.Pi / 2}}),
			},
			z:         cmplx.Exp(complex(0, 0.5)),
			wantedAmp: 1.5,
		},
	} {
		memory := len(tc.conf.Poles)
		if len(tc.conf.Zeros) > memory {
			memory = len(tc.conf.Zeros)
		}
		s := make(signals.Float64Slice, 1000000)
		for i := range s {
			s[i] = real(cmplx.Pow(tc.z, -complex(float64(i), 0)))
		}
		gainBefore := math.Sqrt(stat.Variance(s, nil))

		lti := tc.conf.Make()
		s2 := make(signals.Float64Slice, len(s))
		for i := range s2 {
			s2[i] = real(lti.Next(complex(s[i], 0)))
		}
		gainAfter := math.Sqrt(stat.Variance(s2, nil))

		expectedAmp := cmplx.Abs(tc.conf.H(tc.z))
		if math.Abs(expectedAmp-tc.wantedAmp) > math.Abs(0.03*tc.wantedAmp) {
			t.Errorf("case %v: got expected amp %v, wanted %v", i, expectedAmp, tc.wantedAmp)
		}

		resultingAmp := gainAfter / gainBefore
		if math.Abs(resultingAmp-tc.wantedAmp) > math.Abs(0.03*tc.wantedAmp) {
			t.Errorf("case %v: got resulting amp %v, wanted %v", i, resultingAmp, tc.wantedAmp)
		}
	}
}

func TestDifferenceEquation(t *testing.T) {
	for _, tc := range []struct {
		lti     *LTI
		numbers bool
		want    string
	}{
		{
			lti: LTIConf{
				Gain:  0.0,
				Poles: []complex128{0, 0},
				Zeros: []complex128{0, 0, 0},
			}.Make(),
			numbers: false,
			want:    "(x[n] * g * qc0 + x[n-1] * g * qc1 + x[n-2] * g * qc2 + x[n-3] * g * qc3 - (y[n-2] * pc1 + y[n-3] * pc2)) / pc0",
		},
		{
			lti: LTIConf{
				Gain:  0.5,
				Zeros: []complex128{-1},
				Poles: []complex128{1},
			}.Make(),
			numbers: true,
			want:    "(x[n] * (0.5+0i) + x[n-1] * (0.5+0i) - (y[n-1] * (-1+0i))) / (1+0i)",
		},
	} {
		if got := tc.lti.DifferenceEquation(tc.numbers); got != tc.want {
			t.Errorf("got %q, wanted %q", got, tc.want)
		}
	}
}

func TestHEquation(t *testing.T) {
	want := "g * (z - q0) * (z - q1) * (z - q2) / (z - p0) * (z - p1))"
	got := LTIConf{
		Gain:  0.0,
		Poles: []complex128{0, 0},
		Zeros: []complex128{0, 0, 0},
	}.HEquation(false)
	if got != want {
		t.Errorf("got %q, wanted %q", got, want)
	}
}

func TestH(t *testing.T) {
	for _, tc := range []struct {
		c      LTIConf
		z      complex128
		wanted float64
	}{
		{
			c: LTIConf{
				Gain:  2.0,
				Poles: nil,
				Zeros: []complex128{cmplx.Exp(complex(0, math.Pi/2))},
			},
			z:      cmplx.Exp(complex(0, math.Pi/2)),
			wanted: 0,
		},
		{
			c: LTIConf{
				Gain:  2.0,
				Poles: nil,
				Zeros: []complex128{cmplx.Exp(complex(0, math.Pi/2))},
			},
			z:      cmplx.Exp(complex(0, math.Pi/2+1)),
			wanted: 2.0 * cmplx.Abs(cmplx.Exp(complex(0, math.Pi/2+1))-cmplx.Exp(complex(0, math.Pi/2))),
		},
	} {
		if got := cmplx.Abs(tc.c.H(tc.z)); got != tc.wanted {
			t.Errorf("got %v, wanted %v", got, tc.wanted)
		}
	}
}

func TestCoeffs(t *testing.T) {
	for _, tc := range []struct {
		num  int
		want [][][]int
	}{
		{
			num:  0,
			want: [][][]int{},
		},
		{
			num:  1,
			want: [][][]int{{{0}}},
		},
		{
			num:  3,
			want: [][][]int{{{0}, {1}, {2}}, {{0, 1}, {0, 2}, {1, 2}}, {{0, 1, 2}}},
		},
		{
			num:  4,
			want: [][][]int{{{0}, {1}, {2}, {3}}, {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}, {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}}, {{0, 1, 2, 3}}},
		},
	} {
		pzs := make([]complex128, tc.num)
		for i := range pzs {
			pzs[i] = complex(rand.Float64(), rand.Float64())
		}
		coeffs := coeffs(pzs)
		if coeffs[0] != 1 {
			t.Errorf("got z^0 coeff %v, wanted 1", coeffs[0])
		}
		for i := 0; i < tc.num; i++ {
			want := complex128(0)
			for _, parts := range tc.want[i] {
				prod := complex128(1)
				for _, part := range parts {
					prod *= -pzs[part]
				}
				want += prod
			}
			got := coeffs[1+i]
			if got != want {
				t.Errorf("got coeff %v for z^-%v, wanted %v", got, i+1, want)
			}
		}
	}
}

func TestCombine(t *testing.T) {
	for _, tc := range []struct {
		length int
		num    int
		want   [][]int
	}{
		{
			length: 3,
			num:    0,
			want:   [][]int{},
		},
		{
			length: 3,
			num:    2,
			want:   [][]int{{0, 1}, {2, 1}, {2, 0}},
		},
		{
			length: 3,
			num:    1,
			want:   [][]int{{0}, {2}, {1}},
		},
		{
			length: 4,
			num:    2,
			want:   [][]int{{0, 1}, {0, 2}, {0, 3}, {2, 1}, {3, 1}, {2, 3}},
		},
		{
			length: 4,
			num:    3,
			want:   [][]int{{0, 1, 2}, {0, 1, 3}, {1, 2, 3}, {0, 2, 3}},
		},
	} {
		got := combine(tc.length, tc.num)
		if err := checkEq(got, tc.want); err != nil {
			t.Error(err)
		}
	}
}
