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

// $ go mod init example.com/Reverse8Char/v2
// $ go mod tidy
// $ go build

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

const (
	Alphabet = "abcdefghijklmnopqrstuvwxyz"
	AlphaLen = len(Alphabet)
	StrLen   = 8
	InputDim = StrLen * AlphaLen  // 208
)

type Neuron struct {
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
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

type MLP struct {
	Hidden []*Neuron `json:"hidden"`
	Output []*Neuron `json:"output"`
}

func NewMLP(inputDim, hiddenDim, outputDim int) *MLP {
	m := &MLP{
		Hidden: make([]*Neuron, hiddenDim),
		Output: make([]*Neuron, outputDim),
	}
	for i := range m.Hidden {
		m.Hidden[i] = &Neuron{Weights: randomWeights(inputDim), Bias: rand.Float64()}
	}
	for i := range m.Output {
		m.Output[i] = &Neuron{Weights: randomWeights(hiddenDim), Bias: rand.Float64()}
	}
	return m
}

func randomWeights(n int) []float64 {
	w := make([]float64, n)
	for i := range w { w[i] = rand.NormFloat64() * 0.1 }
	return w
}

func (m *MLP) Train(inputs []float64, targets []float64, lr float64) {
	// 1. Forward Pass
	hOutputs := make([]float64, len(m.Hidden))
	for i, n := range m.Hidden { hOutputs[i] = n.Process(inputs) }

	oOutputs := make([]float64, len(m.Output))
	for i, n := range m.Output { oOutputs[i] = n.Process(hOutputs) }

	// 2. Backprop Output Layer
	oGradients := make([]float64, len(m.Output))
	for i := range m.Output {
		err := targets[i] - oOutputs[i]
		oGradients[i] = err * sigmoidDerivative(oOutputs[i])
	}

	// 3. Backprop Hidden Layer
	hGradients := make([]float64, len(m.Hidden))
	for i := range m.Hidden {
		var err float64
		for j := range m.Output {
			err += oGradients[j] * m.Output[j].Weights[i]
		}
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
}

func (m *MLP) Predict(inputs []float64) string {
	hOutputs := make([]float64, len(m.Hidden))
	for i, n := range m.Hidden { hOutputs[i] = n.Process(inputs) }
	
	res := ""
	for _, n := range m.Output {
		val := n.Process(hOutputs)
		res += string(rune(int(val*255.0 + 0.5)))
	}
	return res
}

func Softmax(logits []float64) []float64 {
	max := logits[0]
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

func stringToFloats(s string) []float64 {
	f := make([]float64, 8)
	for i := 0; i < 8 && i < len(s); i++ { f[i] = float64(s[i]) / 255.0 }
	return f
}

func reverse(s string) string {
	r := []rune(s); for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 { r[i], r[j] = r[j], r[i] }; return string(r)
}

func generateRandomString(n int) string {
	const l = "abcdefghijklmnopqrstuvwxyz"; b := make([]byte, n)
	for i := range b { b[i] = l[rand.Intn(len(l))] }; return string(b)
}

// Helper: Char to index (a=0, b=1...)
func charToIndex(c byte) int {
	for i := 0; i < AlphaLen; i++ {
		if Alphabet[i] == c { return i }
	}
	return 0
}

// One-Hot Encode an 8-char string into a slice of 208 floats
func encode(s string) []float64 {
	vec := make([]float64, InputDim)
	for i := 0; i < StrLen; i++ {
		charIdx := charToIndex(s[i])
		vec[i*AlphaLen+charIdx] = 1.0
	}
	return vec
}

// Decode a slice of 208 floats back to 8-char string
func decode(vec []float64) string {
	res := make([]byte, StrLen)
	for i := 0; i < StrLen; i++ {
		maxVal := -1.0
		bestIdx := 0
		// Find the index with the highest probability in this char's block
		for j := 0; j < AlphaLen; j++ {
			val := vec[i*AlphaLen+j]
			if val > maxVal {
				maxVal = val
				bestIdx = j
			}
		}
		res[i] = Alphabet[bestIdx]
	}
	return string(res)
}

func decodeWithProbabilities(vec []float64) {
	for i := 0; i < StrLen; i++ {
		// Extract the 26-float block for this character
		block := vec[i*AlphaLen : (i+1)*AlphaLen]
		probs := Softmax(block)

		// Find the top result
		bestIdx := 0
		for j, p := range probs {
			if p > probs[bestIdx] { bestIdx = j }
		}

		fmt.Printf("Pos %d: %c (Confidence: %.2f%%)\n",
			i, Alphabet[bestIdx], probs[bestIdx]*100)
	}
}

type Candidate struct {
	Char byte
	Prob float64
}

// DisplayTop3 takes the 208-length output vector from the network
func DisplayTop3(outputVec []float64) {
	fmt.Println("\n--- Top 3 Predictions per Position ---")

	for i := 0; i < StrLen; i++ {
		// 1. Get the 26-float block for this character position
		block := outputVec[i*AlphaLen : (i+1)*AlphaLen]

		// 2. Apply Softmax to get probabilities that sum to 1.0
		probs := Softmax(block)

		// 3. Map probabilities to their characters
		candidates := make([]Candidate, AlphaLen)
		for j := 0; j < AlphaLen; j++ {
			candidates[j] = Candidate{Char: Alphabet[j], Prob: probs[j]}
		}

		// 4. Sort candidates by probability descending
		sort.Slice(candidates, func(a, b int) bool {
			return candidates[a].Prob > candidates[b].Prob
		})

		// 5. Print the top 3
		fmt.Printf("Pos %d: ", i)
		for k := 0; k < 3; k++ {
			fmt.Printf(" [%c: %.1f%%] ", candidates[k].Char, candidates[k].Prob*100)
		}
		fmt.Println()
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// MLP structure: 208 inputs -> 64 hidden -> 208 outputs
	model := NewMLP(InputDim, 64, InputDim)

	fmt.Println("Training with One-Hot Encoding (this may take a minute)...")
	for e := 0; e < 150000; e++ {
		s := generateRandomString(StrLen)
		input := encode(s)
		target := encode(reverse(s))
		model.Train(input, target, 0.1)

		if e%30000 == 0 { fmt.Printf("Progress: %d%%\n", (e*100)/150000) }
	}

	test := "onionsys"
	// Get raw float output from model
	hOutputs := make([]float64, len(model.Hidden))
	for i, n := range model.Hidden { hOutputs[i] = n.Process(encode(test)) }

	rawPrediction := make([]float64, InputDim)
	for i, n := range model.Output { rawPrediction[i] = n.Process(hOutputs) }

	DisplayTop3(rawPrediction)

	fmt.Printf("\nInput:  %s\nTarget: %s\nResult: %s\n", test, reverse(test), decode(rawPrediction))
}
