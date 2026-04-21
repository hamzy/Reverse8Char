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

/*
 * This work is marked with CC0 1.0 Universal.
 * http://creativecommons.org
 */

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
	InputDim = StrLen * AlphaLen // 208
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

func sigmoid(x float64) float64           { return 1.0 / (1.0 + math.Exp(-math.Max(-20, math.Min(20, x)))) }
func sigmoidDerivative(s float64) float64  { return s * (1.0 - s) }

// --- CORE LOGIC ---

func (n *Neuron) Process(inputs []float64) float64 {
	var sum float64
	for i, v := range inputs { sum += v * n.Weights[i] }
	return sigmoid(sum + n.Bias)
}

func NewMLP(inputDim, hiddenDim, outputDim int) *MLP {
	m := &MLP{
		Hidden: make([]*Neuron, hiddenDim),
		Output: make([]*Neuron, outputDim),
	}
	for i := range m.Hidden {
		m.Hidden[i] = &Neuron{Weights: randomWeightsXavier(inputDim, hiddenDim), Bias: 0}
	}
	for i := range m.Output {
		m.Output[i] = &Neuron{Weights: randomWeightsXavier(hiddenDim, outputDim), Bias: 0}
	}
	return m
}

func Softmax(logits []float64) []float64 {
	max := -math.MaxFloat64
	for _, v := range logits { if v > max { max = v } }
	res := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		res[i] = math.Exp(v - max)
		sum += res[i]
	}
	for i := range res { res[i] /= (sum + 1e-15) }
	return res
}

func (m *MLP) TrainCE(inputs []float64, targets []float64, lr float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// 1. Forward Pass
	hSums := make([]float64, len(m.Hidden))
	hActs := make([]float64, len(m.Hidden))
	for i, n := range m.Hidden {
		hSums[i] = n.Process(inputs)
		hActs[i] = sigmoid(hSums[i])
	}

	oSums := make([]float64, len(m.Output))
	for i, n := range m.Output { oSums[i] = n.Process(hActs) }

	// 2. Output Gradients (Softmax + CE)
	// Key: The derivative of Softmax+CE is simply (target - prediction)
	oGradients := make([]float64, len(m.Output))
	for i := 0; i < StrLen; i++ {
		start, end := i*AlphaLen, (i+1)*AlphaLen
		probs := Softmax(oSums[start:end])
		for j := 0; j < AlphaLen; j++ {
			idx := start + j
			oGradients[idx] = targets[idx] - probs[j]
		}
	}

	// 3. Hidden Gradients (Backprop)
	hGradients := make([]float64, len(m.Hidden))
	for i := range m.Hidden {
		var err float64
		for j := range m.Output { err += oGradients[j] * m.Output[j].Weights[i] }
		hGradients[i] = err * sigmoidDerivative(hActs[i])
	}

	// 4. Update Weights
	for i, n := range m.Output {
		for j := range n.Weights { n.Weights[j] += hActs[j] * oGradients[i] * lr }
		n.Bias += oGradients[i] * lr
	}
	for i, n := range m.Hidden {
		for j := range n.Weights { n.Weights[j] += inputs[j] * hGradients[i] * lr }
		n.Bias += hGradients[i] * lr
	}
}

// Prediction Logic Fix: Must re-run forward pass correctly
func (m *MLP) Predict(s string) string {
	inputs := encode(s)
	h := make([]float64, len(m.Hidden))
	for i, n := range m.Hidden { h[i] = sigmoid(n.Process(inputs)) }

	res := ""
	for i := 0; i < StrLen; i++ {
		block := make([]float64, AlphaLen)
		for j := 0; j < AlphaLen; j++ {
			block[j] = m.Output[i*AlphaLen+j].Process(h)
		}
		probs := Softmax(block)
		best := 0
		for idx, p := range probs { if p > probs[best] { best = idx } }
		res += string(Alphabet[best])
	}
	return res
}

// --- DATA HELPERS ---

func encode(s string) []float64 {
	v := make([]float64, InputDim)
	for i := 0; i < StrLen; i++ {
		idx := -1
		for j := range Alphabet {
			if Alphabet[j] == s[i] { idx = j; break }
		}
		if idx != -1 { v[i*AlphaLen+idx] = 1.0 }
	}
	return v
}

func reverse(s string) string {
	r := []rune(s)
	for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 { r[i], r[j] = r[j], r[i] }
	return string(r)
}

func generateRandomString(n int) string {
	b := make([]byte, n)
	for i := range b { b[i] = Alphabet[rand.Intn(AlphaLen)] }
	return string(b)
}

// --- STABILITY UTILITIES ---

// Xavier Initialization prevents exploding/vanishing gradients
func randomWeightsXavier(fanIn, fanOut int) []float64 {
	limit := math.Sqrt(6.0 / float64(fanIn+fanOut))
	w := make([]float64, fanIn)
	for i := range w {
		w[i] = (rand.Float64()*2 - 1) * limit
	}
	return w
}

func main() {
	rand.Seed(time.Now().UnixNano())
	model := NewMLP(InputDim, 128, InputDim)

	fmt.Println("Training...")
	for e := 1; e <= 200000; e++ {
		s := generateRandomString(StrLen)
		model.TrainCE(encode(s), encode(reverse(s)), 0.1)

		if e%40000 == 0 {
			test := "onionsys"
			fmt.Printf("Step %d | Test: %s -> %s\n", e, test, model.Predict(test))
		}
	}
}

// ... (Use previously provided encode, reverse, generateRandomString, and NewMLP helpers)
