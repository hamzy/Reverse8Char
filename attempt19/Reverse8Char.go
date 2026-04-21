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

type AdamState struct {
	M []float64
	V []float64
}

type Neuron struct {
	Weights []float64
	Bias    float64
	WState  AdamState
	BState  AdamState
}

type MLP struct {
	Hidden []*Neuron
	Output []*Neuron
	Steps  int
	mu     sync.Mutex
}

// --- MATH & UTILITIES ---

func sigmoid(x float64) float64      { return 1.0 / (1.0 + math.Exp(-math.Max(-20, math.Min(20, x)))) }
func sigmoidDeriv(s float64) float64 { return s * (1.0 - s) }

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

func NewNeuron(inDim int) *Neuron {
	limit := math.Sqrt(6.0 / float64(inDim+1))
	w := make([]float64, inDim)
	for i := range w { w[i] = (rand.Float64()*2 - 1) * limit }
	return &Neuron{
		Weights: w,
		WState:  AdamState{M: make([]float64, inDim), V: make([]float64, inDim)},
		BState:  AdamState{M: make([]float64, 1), V: make([]float64, 1)},
	}
}

func NewMLP(in, hid, out int) *MLP {
	m := &MLP{Hidden: make([]*Neuron, hid), Output: make([]*Neuron, out)}
	for i := range m.Hidden { m.Hidden[i] = NewNeuron(in) }
	for i := range m.Output { m.Output[i] = NewNeuron(hid) }
	return m
}

// --- ADAM TRAINING LOGIC ---

func (m *MLP) TrainAdam(inputs, targets []float64, lr float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Steps++
	t := float64(m.Steps)

	const beta1, beta2, eps = 0.9, 0.999, 1e-8

	// 1. Forward Pass
	hActs := make([]float64, len(m.Hidden))
	for i, n := range m.Hidden {
		var sum float64
		for j, v := range inputs { sum += v * n.Weights[j] }
		hActs[i] = sigmoid(sum + n.Bias)
	}
	oSums := make([]float64, len(m.Output))
	for i, n := range m.Output {
		var sum float64
		for j, v := range hActs { sum += v * n.Weights[j] }
		oSums[i] = sum + n.Bias
	}

	// 2. Gradients
	oGrads := make([]float64, len(m.Output))
	for i := 0; i < StrLen; i++ {
		probs := Softmax(oSums[i*AlphaLen : (i+1)*AlphaLen])
		for j := 0; j < AlphaLen; j++ {
			idx := i*AlphaLen + j
			oGrads[idx] = probs[j] - targets[idx]
		}
	}

	hGrads := make([]float64, len(m.Hidden))
	for i := range m.Hidden {
		var err float64
		for j := range m.Output { err += oGrads[j] * m.Output[j].Weights[i] }
		hGrads[i] = err * sigmoidDeriv(hActs[i])
	}

	// 3. Apply Adam Updates
	// Output Layer
	for i, n := range m.Output {
		for j := range n.Weights {
			g := oGrads[i] * hActs[j]
			n.WState.M[j] = beta1*n.WState.M[j] + (1-beta1)*g
			n.WState.V[j] = beta2*n.WState.V[j] + (1-beta2)*g*g
			mHat := n.WState.M[j] / (1 - math.Pow(beta1, t))
			vHat := n.WState.V[j] / (1 - math.Pow(beta2, t))
			n.Weights[j] -= lr * mHat / (math.Sqrt(vHat) + eps)
		}
		// Bias
		bg := oGrads[i]
		n.BState.M[0] = beta1*n.BState.M[0] + (1-beta1)*bg
		n.BState.V[0] = beta2*n.BState.V[0] + (1-beta2)*bg*bg
		mHat := n.BState.M[0] / (1 - math.Pow(beta1, t))
		vHat := n.BState.V[0] / (1 - math.Pow(beta2, t))
		n.Bias -= lr * mHat / (math.Sqrt(vHat) + eps)
	}

	// Hidden Layer
	for i, n := range m.Hidden {
		for j := range n.Weights {
			g := hGrads[i] * inputs[j]
			n.WState.M[j] = beta1*n.WState.M[j] + (1-beta1)*g
			n.WState.V[j] = beta2*n.WState.V[j] + (1-beta2)*g*g
			mHat := n.WState.M[j] / (1 - math.Pow(beta1, t))
			vHat := n.WState.V[j] / (1 - math.Pow(beta2, t))
			n.Weights[j] -= lr * mHat / (math.Sqrt(vHat) + eps)
		}
		// Bias
		bg := hGrads[i]
		n.BState.M[0] = beta1*n.BState.M[0] + (1-beta1)*bg
		n.BState.V[0] = beta2*n.BState.V[0] + (1-beta2)*bg*bg
		mHat := n.BState.M[0] / (1 - math.Pow(beta1, t))
		vHat := n.BState.V[0] / (1 - math.Pow(beta2, t))
		n.Bias -= lr * mHat / (math.Sqrt(vHat) + eps)
	}
}

// --- PREDICTION & HELPERS ---

func (m *MLP) Predict(s string) string {
	in := encode(s)
	h := make([]float64, len(m.Hidden))
	for i, n := range m.Hidden {
		var sum float64
		for j, v := range in { sum += v * n.Weights[j] }
		h[i] = sigmoid(sum + n.Bias)
	}
	res := ""
	for i := 0; i < StrLen; i++ {
		logits := make([]float64, AlphaLen)
		for j := 0; j < AlphaLen; j++ {
			var sum float64
			for k, v := range h { sum += v * m.Output[i*AlphaLen+j].Weights[k] }
			logits[j] = sum + m.Output[i*AlphaLen+j].Bias
		}
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
	r := []rune(s)
	for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 { r[i], r[j] = r[j], r[i] }
	return string(r)
}

func main() {
	rand.Seed(time.Now().UnixNano())
	model := NewMLP(InputDim, 128, InputDim)

	fmt.Println("Training with Adam...")
	start := time.Now()
	for e := 1; e <= 100000; e++ {
		s := ""
		for l := 0; l < 8; l++ { s += string(Alphabet[rand.Intn(26)]) }
		model.TrainAdam(encode(s), encode(reverse(s)), 0.001)

		if e%20000 == 0 {
			test := "onionsys"
			fmt.Printf("Step %d | %s -> %s\n", e, test, model.Predict(test))
			if model.Predict(test) == "sysnoino" {
				fmt.Println("Success!")
				break
			}
		}
	}
	fmt.Printf("\nDone in %v\nFinal: onionsys -> %s\n", time.Since(start), model.Predict("onionsys"))
}
