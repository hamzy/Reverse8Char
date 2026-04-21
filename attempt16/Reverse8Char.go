/*
 * This work is marked with CC0 1.0 Universal. 
 * To view a copy of this license, visit:
 * http://creativecommons.org
 *
 * No Copyright: The person who associated a work with this deed has dedicated 
 * the work to the public domain by waiving all of his or her rights to the work 
 * worldwide under copyright law, including all related and neighboring rights, 
 * to the extent allowed by law.
 */

// https://share.google/aimode/7VvT09F5IOnQtqu7Z

// $ go mod init example.com/Reverse8Char/v2
// $ go mod tidy
// $ go build

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

const (
	Alphabet = "abcdefghijklmnopqrstuvwxyz"
	AlphaLen = len(Alphabet)
	StrLen   = 8
	InputDim = StrLen * AlphaLen
)

type Neuron struct {
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
}

type MLP struct {
	Hidden []*Neuron  `json:"hidden"`
	Output []*Neuron  `json:"output"`
	mu     sync.Mutex
}

func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-math.Max(-20, math.Min(20, x)))) }
func sigmoidDeriv(s float64) float64 { return s * (1.0 - s) }

func (n *Neuron) Process(inputs []float64) float64 {
	var sum float64
	for i, v := range inputs { sum += v * n.Weights[i] }
	return sum + n.Bias
}

func Softmax(logits []float64) []float64 {
	max := -math.MaxFloat64
	for _, v := range logits { if v > max { max = v } }
	res, sum := make([]float64, len(logits)), 0.0
	for i, v := range logits {
		res[i] = math.Exp(v - max)
		sum += res[i]
	}
	for i := range res { res[i] /= (sum + 1e-15) }
	return res
}

func NewMLP(in, hid, out int) *MLP {
	m := &MLP{Hidden: make([]*Neuron, hid), Output: make([]*Neuron, out)}
	limitH := math.Sqrt(6.0 / float64(in+hid))
	for i := range m.Hidden {
		w := make([]float64, in)
		for j := range w { w[j] = (rand.Float64()*2 - 1) * limitH }
		m.Hidden[i] = &Neuron{Weights: w}
	}
	limitO := math.Sqrt(6.0 / float64(hid+out))
	for i := range m.Output {
		w := make([]float64, hid)
		for j := range w { w[j] = (rand.Float64()*2 - 1) * limitO }
		m.Output[i] = &Neuron{Weights: w}
	}
	return m
}

func (m *MLP) Train(inputs, targets []float64, lr float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// 1. Forward
	hActs := make([]float64, len(m.Hidden))
	for i, n := range m.Hidden { hActs[i] = sigmoid(n.Process(inputs)) }
	oSums := make([]float64, len(m.Output))
	for i, n := range m.Output { oSums[i] = n.Process(hActs) }

	// 2. Gradients (Prediction - Target)
	oGrads := make([]float64, len(m.Output))
	for i := 0; i < StrLen; i++ {
		probs := Softmax(oSums[i*AlphaLen : (i+1)*AlphaLen])
		for j := 0; j < AlphaLen; j++ {
			idx := i*AlphaLen + j
			oGrads[idx] = probs[j] - targets[idx] // Correct Gradient for CE
		}
	}

	hGrads := make([]float64, len(m.Hidden))
	for i := range m.Hidden {
		var err float64
		for j := range m.Output { err += oGrads[j] * m.Output[j].Weights[i] }
		hGrads[i] = err * sigmoidDeriv(hActs[i])
	}

	// 3. Update
	for i, n := range m.Output {
		for j := range n.Weights { n.Weights[j] -= hActs[j] * oGrads[i] * lr }
		n.Bias -= oGrads[i] * lr
	}
	for i, n := range m.Hidden {
		for j := range n.Weights { n.Weights[j] -= inputs[j] * hGrads[i] * lr }
		n.Bias -= hGrads[i] * lr
	}
}

func (m *MLP) Predict(s string) string {
	in := encode(s)
	h := make([]float64, len(m.Hidden))
	for i, n := range m.Hidden { h[i] = sigmoid(n.Process(in)) }
	res := ""
	for i := 0; i < StrLen; i++ {
		logits := make([]float64, AlphaLen)
		for j := 0; j < AlphaLen; j++ { logits[j] = m.Output[i*AlphaLen+j].Process(h) }
		p := Softmax(logits)
		best := 0
		for idx, val := range p { if val > p[best] { best = idx } }
		res += string(Alphabet[best])
	}
	return res
}

func encode(s string) []float64 {
	v := make([]float64, InputDim)
	for i := 0; i < len(s) && i < StrLen; i++ {
		for j, char := range Alphabet {
			if byte(char) == s[i] { v[i*AlphaLen+j] = 1.0; break }
		}
	}
	return v
}

func reverse(s string) string {
	r := []rune(s); for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 { r[i], r[j] = r[j], r[i] }; return string(r)
}

func main() {
	rand.Seed(time.Now().UnixNano())
	model := NewMLP(InputDim, 128, InputDim)
	start := time.Now()

	fmt.Println("Training...")
	for e := 0; e <= 100000; e++ {
		s := ""
		for l := 0; l < 8; l++ { s += string(Alphabet[rand.Intn(26)]) }
		lr := 0.1 * math.Pow(0.9, float64(e/20000))
		model.Train(encode(s), encode(reverse(s)), lr)

		if e%20000 == 0 {
			test := "googleai" // "onionsys"
			fmt.Printf("Step %d | %s -> %s\n", e, test, model.Predict(test))
		}
	}
	fmt.Printf("Time: %v\n", time.Since(start))
}
