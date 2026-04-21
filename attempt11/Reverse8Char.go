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
	"sort"
	"sync"
	"time"
)

const (
	Alphabet = "abcdefghijklmnopqrstuvwxyz"
	AlphaLen = len(Alphabet)
	StrLen   = 8
	InputDim = StrLen * AlphaLen // 208 dimensions
)

type Neuron struct {
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
}

type MLP struct {
	Hidden []*Neuron `json:"hidden"`
	Output []*Neuron `json:"output"`
	mu     sync.Mutex // Protects weights during parallel updates
}

type Candidate struct {
	Char byte
	Prob float64
}

func sigmoid(x float64) float64           { return 1.0 / (1.0 + math.Exp(-x)) }
func sigmoidDerivative(s float64) float64  { return s * (1.0 - s) }

func (n *Neuron) Process(inputs []float64) float64 {
	var sum float64
	for i, v := range inputs {
		sum += v * n.Weights[i]
	}
	return sigmoid(sum + n.Bias)
}

func NewMLP(inputDim, hiddenDim, outputDim int) *MLP {
	m := &MLP{
		Hidden: make([]*Neuron, hiddenDim),
		Output: make([]*Neuron, outputDim),
	}
	for i := range m.Hidden {
		m.Hidden[i] = &Neuron{Weights: randomWeights(inputDim), Bias: rand.NormFloat64() * 0.1}
	}
	for i := range m.Output {
		m.Output[i] = &Neuron{Weights: randomWeights(hiddenDim), Bias: rand.NormFloat64() * 0.1}
	}
	return m
}

func randomWeights(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = rand.NormFloat64() * 0.1
	}
	return w
}

func Softmax(logits []float64) []float64 {
	var max float64 = -math.MaxFloat64
	for _, v := range logits {
		if v > max { max = v }
	}
	res := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		res[i] = math.Exp(v - max)
		sum += res[i]
	}
	for i := range res {
		res[i] /= sum
	}
	return res
}

func (m *MLP) TrainCE(inputs []float64, targets []float64, lr float64) float64 {
	m.mu.Lock()
	defer m.mu.Unlock()

	// 1. Forward Pass
	hOutputs := make([]float64, len(m.Hidden))
	for i, n := range m.Hidden { hOutputs[i] = n.Process(inputs) }

	rawOutputs := make([]float64, len(m.Output))
	for i, n := range m.Output { rawOutputs[i] = n.Process(hOutputs) }

	// 2. Output Gradients (Softmax + CE)
	oGradients := make([]float64, len(m.Output))
	var totalLoss float64
	for i := 0; i < StrLen; i++ {
		start, end := i*AlphaLen, (i+1)*AlphaLen
		// Corrected: probs is local block of 26
		probs := Softmax(rawOutputs[start:end])
		for j := 0; j < AlphaLen; j++ {
			idx := start + j
			// idx is global (0-207), j is local (0-25)
			oGradients[idx] = targets[idx] - probs[j]
			if targets[idx] > 0.5 {
				totalLoss -= math.Log(math.Max(probs[j], 1e-15))
			}
		}
	}

	// 3. Hidden Gradients
	hGradients := make([]float64, len(m.Hidden))
	for i := range m.Hidden {
		var err float64
		for j := range m.Output { err += oGradients[j] * m.Output[j].Weights[i] }
		hGradients[i] = err * sigmoidDerivative(hOutputs[i])
	}

	// 4. Update Weights
	for i, n := range m.Output {
		for j := range n.Weights { n.Weights[j] += hOutputs[j] * oGradients[i] * lr }
		n.Bias += oGradients[i] * lr
	}
	for i, n := range m.Hidden {
		for j := range n.Weights { n.Weights[j] += inputs[j] * hGradients[i] * lr }
		n.Bias += hGradients[i] * lr
	}
	return totalLoss
}

func (m *MLP) PredictRaw(inputs []float64) []float64 {
	hOutputs := make([]float64, len(m.Hidden))
	for i, n := range m.Hidden { hOutputs[i] = n.Process(inputs) }
	res := make([]float64, len(m.Output))
	for i, n := range m.Output { res[i] = n.Process(hOutputs) }
	return res
}

func charToIndex(c byte) int {
	for i := 0; i < AlphaLen; i++ {
		if Alphabet[i] == c { return i }
	}
	return 0
}

func encode(s string) []float64 {
	vec := make([]float64, InputDim)
	for i := 0; i < StrLen; i++ {
		vec[i*AlphaLen+charToIndex(s[i])] = 1.0
	}
	return vec
}

func reverse(s string) string {
	r := []rune(s)
	for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 { r[i], r[j] = r[j], r[i] }
	return string(r)
}

func generateRandomString(n int) string {
	l := "abcdefghijklmnopqrstuvwxyz"
	b := make([]byte, n)
	for i := range b { b[i] = l[rand.Intn(len(l))] }
	return string(b)
}

func DisplayTop3(outputVec []float64) {
	fmt.Println("\n--- Top 3 Predictions per Position ---")
	for i := 0; i < StrLen; i++ {
		block := outputVec[i*AlphaLen : (i+1)*AlphaLen]
		probs := Softmax(block)
		candidates := make([]Candidate, AlphaLen)
		for j := 0; j < AlphaLen; j++ {
			candidates[j] = Candidate{Char: Alphabet[j], Prob: probs[j]}
		}
		sort.Slice(candidates, func(a, b int) bool { return candidates[a].Prob > candidates[b].Prob })
		fmt.Printf("Pos %d: ", i)
		for k := 0; k < 3; k++ {
			fmt.Printf(" [%c: %.1f%%] ", candidates[k].Char, candidates[k].Prob*100)
		}
		fmt.Println()
	}
}

type Job struct {
	Input  []float64
	Target []float64
	LR     float64
}

func main() {
	rand.Seed(time.Now().UnixNano())
	const file = "decay_weights.json"
	var model *MLP

	if _, err := os.Stat(file); err == nil {
		fmt.Println("Loading weights...")
		data, _ := os.ReadFile(file)
		json.Unmarshal(data, &model)
	} else {
		model = NewMLP(InputDim, 128, InputDim)
		const numWorkers = 8
		const totalSamples = 150000
		jobs := make(chan Job, 100)
		var wg sync.WaitGroup

		// 1. Launch Worker Pool
		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := range jobs {
					model.TrainCE(j.Input, j.Target, j.LR)
				}
			}()
		}

		// 2. Feed Jobs with Decay
		start := time.Now()
		initialLR := 0.2
		for i := 0; i < totalSamples; i++ {
			s := generateRandomString(StrLen)
			// Decay the learning rate every 15,000 samples
			currentLR := initialLR * math.Pow(0.9, float64(i/15000))

			jobs <- Job{
				Input:  encode(s),
				Target: encode(reverse(s)),
				LR:     currentLR,
			}

			if i%30000 == 0 {
				fmt.Printf("Step %d | Current LR: %.4f\n", i, currentLR)
			}
		}
		close(jobs)
		wg.Wait()
		fmt.Printf("Done in %v\n", time.Since(start))

		d, _ := json.Marshal(model)
		os.WriteFile(file, d, 0644)
	}

	test := "onionsys"
	prediction := model.PredictRaw(encode(test))

	fmt.Printf("\nInput:  %s", test)
	fmt.Printf("\nTarget: %s", reverse(test))
	DisplayTop3(prediction)
}
