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
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
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

type Candidate struct {
	Char byte
	Prob float64
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

func clip(g float64) float64 {
	const threshold = 1.0
	if g > threshold { return threshold }
	if g < -threshold { return -threshold }
	return g
}

func sigmoid(x float64) float64 {
	if x > 20 { return 1.0 } // Prevent overflow
	if x < -20 { return 0.0 }
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(s float64) float64 { return s * (1.0 - s) }

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
	var max float64 = -math.MaxFloat64
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
	hOutputs := make([]float64, len(m.Hidden))
	for i, n := range m.Hidden { hOutputs[i] = n.Process(inputs) }
	
	rawOutputs := make([]float64, len(m.Output))
	for i, n := range m.Output { rawOutputs[i] = n.Process(hOutputs) }

	// 2. Gradients (Softmax + Cross-Entropy)
	oGradients := make([]float64, len(m.Output))
	for i := 0; i < StrLen; i++ {
		start, end := i*AlphaLen, (i+1)*AlphaLen
		probs := Softmax(rawOutputs[start:end])
		for j := 0; j < AlphaLen; j++ {
			idx := start + j
			oGradients[idx] = clip(targets[idx] - probs[j])
		}
	}

	hGradients := make([]float64, len(m.Hidden))
	for i := range m.Hidden {
		var err float64
		for j := range m.Output { err += oGradients[j] * m.Output[j].Weights[i] }
		hGradients[i] = clip(err * sigmoidDerivative(hOutputs[i]))
	}

	// 3. Update Weights
	for i, n := range m.Output {
		for j := range n.Weights { n.Weights[j] += hOutputs[j] * oGradients[i] * lr }
		n.Bias += oGradients[i] * lr
	}
	for i, n := range m.Hidden {
		for j := range n.Weights { n.Weights[j] += inputs[j] * hGradients[i] * lr }
		n.Bias += hGradients[i] * lr
	}
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

func (m *MLP) Validate(count int) float64 {
	correct := 0
	for i := 0; i < count; i++ {
		test := generateRandomString(StrLen)
		target := reverse(test)
		
		h := make([]float64, len(m.Hidden))
		for i, n := range m.Hidden { h[i] = n.Process(encode(test)) }
		
		match := true
		for pos := 0; pos < StrLen; pos++ {
			best, max := 0, -1.0
			for j := 0; j < AlphaLen; j++ {
				val := m.Output[pos*AlphaLen+j].Process(h)
				if val > max { max = val; best = j }
			}
			if Alphabet[best] != target[pos] { match = false; break }
		}
		if match { correct++ }
	}
	return (float64(correct) / float64(count)) * 100
}

func main() {
	rand.Seed(time.Now().UnixNano())
	const file = "stable_weights.json"
	var model *MLP

	if _, err := os.Stat(file); err == nil {
		fmt.Println("Loading weights...")
		d, _ := os.ReadFile(file)
		json.Unmarshal(d, &model)
	} else {
		model = NewMLP(InputDim, 128, InputDim)
		const workers, total = 8, 300000
		var wg sync.WaitGroup
		jobs := make(chan bool, 100)

		// Workers
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for range jobs {
					s := generateRandomString(8)
					model.TrainCE(encode(s), encode(reverse(s)), 0.1)
				}
			}()
		}

		fmt.Println("Training stable model...")
		for i := 1; i <= total; i++ {
			jobs <- true
			if i%50000 == 0 {
				fmt.Printf("Step %d | Accuracy: %.1f%%\n", i, model.Validate(50))
			}
		}
		close(jobs)
		wg.Wait()
		
		d, _ := json.Marshal(model)
		os.WriteFile(file, d, 0644)
	}

	test := "onionsys"
	fmt.Printf("\nInput:  %s\nTarget: %s\nAccuracy: %.1f%%\n", test, reverse(test), model.Validate(100))
}
