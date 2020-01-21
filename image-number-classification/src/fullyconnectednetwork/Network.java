package fullyconnectednetwork;

import trainset.TrainSet;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by Luecx on 30.05.2017.
 */

public class Network {

    private double[][][] weights;
    private double[][] bias;

    private double[][] output;

    private double[][] error_signal;
    private double[][] output_derivative;

    private final int[] NETWORK_LAYER_SIZES;
    private final int   INPUT_SIZE;
    private final int   OUTPUT_SIZE;
    private final int   NETWORK_SIZE;

    private final double randomBiasLowerBound = -0.5;
    private final double randomBiasUpperBound = 0.7;
    private final double randomWeightLowerBound = -1;
    private final double randomWeightUpperBound = 1;

    public Network(int... NETWORK_LAYER_SIZES) {
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE-1];

        this.output = new double[NETWORK_SIZE][];

        this.error_signal = new double[NETWORK_SIZE][];
        this.output_derivative = new double[NETWORK_SIZE][];

        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];

        Random rnd = new Random(123);

        for(int i = 0; i < NETWORK_SIZE; i++) {
            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.output_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];

            this.bias[i] = NetworkTools.createRandomArray(rnd, NETWORK_LAYER_SIZES[i], randomBiasLowerBound, randomBiasUpperBound);

            if(i > 0) {
                weights[i] = NetworkTools.createRandomArray(rnd, NETWORK_LAYER_SIZES[i],NETWORK_LAYER_SIZES[i-1], randomWeightLowerBound, randomWeightUpperBound);
            }
        }
    }


    public double[] calculate(double... input) {
        if(input.length != this.INPUT_SIZE) return null;
        this.output[0] = input;
        for(int layer = 1; layer < NETWORK_SIZE; layer ++) {
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron ++) {

                double sum = bias[layer][neuron];
                for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron ++) {
                    sum += output[layer-1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }
                output[layer][neuron] = sigmoid(sum);
                output_derivative[layer][neuron] = output[layer][neuron] * (1 - output[layer][neuron]);
            }
        }
        return output[NETWORK_SIZE-1];
    }

    private final int batch_size = 100;

    public void train(TrainSet set, int loops) {
        if(set.INPUT_SIZE != INPUT_SIZE || set.OUTPUT_SIZE != OUTPUT_SIZE) return;
        for(int i = 0; i < loops; i++) {
            TrainSet batch = set.extractBatch(batch_size);
            for(int b = 0; b < batch_size; b++) {
                this.train(batch.getInput(b), batch.getOutput(b), 0.3);
            }
            //System.out.println(MSE(batch));
        }
    }

    public double MSE(double[] input, double[] target) {
        if(input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) return 0;
        calculate(input);
        double v = 0;
        for(int i = 0; i < target.length; i++) {
            v += (target[i] - output[NETWORK_SIZE-1][i]) * (target[i] - output[NETWORK_SIZE-1][i]);
        }
        return v / (2d * target.length);
    }

    public double MSE(TrainSet set) {
        double v = 0;
        for(int i = 0; i< set.size(); i++) {
            v += MSE(set.getInput(i), set.getOutput(i));
        }
        return v / set.size();
    }

    public void train(double[] input, double[] target, double eta) {
        if(input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) return;
        calculate(input);
        backpropError(target);
        updateWeights(eta);
    }

    public void backpropError(double[] target) {
        for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE-1]; neuron ++) {
            error_signal[NETWORK_SIZE-1][neuron] = (output[NETWORK_SIZE-1][neuron] - target[neuron])
                    * output_derivative[NETWORK_SIZE-1][neuron];
        }
        for(int layer = NETWORK_SIZE-2; layer > 0; layer --) {
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron ++){
                double sum = 0;
                for(int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer+1]; nextNeuron ++) {
                    sum += weights[layer + 1][nextNeuron][neuron] * error_signal[layer + 1][nextNeuron];
                }
                this.error_signal[layer][neuron] = sum * output_derivative[layer][neuron];
            }
        }
    }

    public void updateWeights(double eta) {
        for(int layer = 1; layer < NETWORK_SIZE; layer++) {
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {

                double delta = - eta * error_signal[layer][neuron];
                bias[layer][neuron] += delta;

                for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron ++) {
                    weights[layer][neuron][prevNeuron] += delta * output[layer-1][prevNeuron];
                }
            }
        }
    }

    private double sigmoid( double x) {
        return 1d / ( 1 + Math.exp(-x));
    }

}
