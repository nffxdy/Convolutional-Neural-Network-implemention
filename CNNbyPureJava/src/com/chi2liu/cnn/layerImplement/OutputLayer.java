/**
 * @projectName: CNNbyPureJava
 * @packageName: com.chi2liu.cnn.layerImplement
 * @fileName: OutputLayer.java
 */
package com.chi2liu.cnn.layerImplement;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Consumer;
import java.util.function.Function;

import com.chi2liu.cnn.cnn.CNN;
import com.chi2liu.cnn.data.DataSet;
import com.chi2liu.cnn.data.DataSet.Record;
import com.chi2liu.cnn.layerInterface.Layer;
import com.chi2liu.cnn.layerInterface.LayerWithKernel;
import com.chi2liu.cnn.util.MathUtil;
import com.chi2liu.cnn.util.Size;


/**
 * @author: chi633 
 * @Email: cliu_whu@yeah.net
 * @date��2017��12��10�� ����6:13:04
 * @className: OutputLayer.java
 * @version: 1.0
 */
public class OutputLayer implements LayerWithKernel {
	
	private final Size size;
    private final double[] bias;
    private final double[][][][] errors;
    private final int batchSize;
    private final int outCount;
    private final Layer preLayer;
    private final double[][][][] outs;
    private double[][][][] kernels;


    public OutputLayer(int batchSize, int classNum, Layer preLayer) {
        this.batchSize = batchSize;
        this.outCount = classNum;
        this.preLayer = preLayer;
        this.size = new Size(1, 1);
        Size kernelSize = preLayer.getSize();
        this.kernels = new double[preLayer.getOutCount()][classNum][kernelSize.getWidth()][kernelSize.getHeight()];
        //Kernels��ʼ��Ϊ�����?
        Random random = new Random();
        for (int i = 0; i < kernels.length; i++) {
            for (int j = 0; j < kernels[0].length; j++) {
                for (int k = 0; k < kernels[0][0].length; k++) {
                    for (int l = 0; l < kernels[0][0][0].length; l++) {
                        kernels[i][j][k][l] = (random.nextDouble() - 0.5) / 10;
                    }
                }
            }
        }
        bias = new double[classNum];
        errors = new double[batchSize][outCount][size.getWidth()][size.getHeight()];
        outs = new double[batchSize][outCount][size.getWidth()][size.getHeight()];
    }

    @Override
    public Size getSize() {
        return null;
    }

    @Override
    public int getOutCount() {
        return outCount;
    }

	@Override
	public void forEachOutput(int recordIndex, Function<Integer, Consumer<double[][]>> function) {
		for (int i = 0; i < outCount; i++) {
            function.apply(i).accept(outs[recordIndex][i]);
        }

	}

	@Override
	public void setNextLayer(Layer layer) {
	}

	@Override
    public double[][] getError(int recordIndex, Integer outIndex) {
        return errors[recordIndex][outIndex];
    }

    @Override
    public double[][] getOut(int recordIndex, int outIndex) {
        return outs[recordIndex][outIndex];
    }

	@Override
	public void forward(Record record) {
		setOutput(record.getIndex(), record, preLayer);
	}

	@Override
	public boolean backPropagation(Record record) {
		return setErrors(record.getIndex(), record);
	}

	@Override
    public double[][] getKernel(int preLayerOutIndex, int thisLayerOutIndex) {
        return kernels[preLayerOutIndex][thisLayerOutIndex];
    }

	@Override
	public void updateKernels() {
		for (int j = 0; j < outCount; j++) {
            for (int i = 0; i < preLayer.getOutCount(); i++) {
                // ��batch��ÿ����¼delta���
                double[][] deltaKernel = null;
                for (int r = 0; r < batchSize; r++) {
                    double[][] error = errors[r][j];
                    if (deltaKernel == null) {
                        deltaKernel = MathUtil.validConvolutional(preLayer.getOut(r, i), error);
                    } else {// �ۻ����
                        deltaKernel = MathUtil.trans(
                                deltaKernel,
                                MathUtil.validConvolutional(preLayer.getOut(r, i), error),
                                v1 -> v2 -> v1 + v2);
                    }
                }
                deltaKernel = MathUtil.trans(deltaKernel, v -> v / batchSize);
                // ���¾����
                double[][] kernel = kernels[i][j];
                deltaKernel = MathUtil.trans(MathUtil.trans(kernel, v -> v * (1 - CNN.LAMBDA * CNN.ALPHA))
                        , MathUtil.trans(deltaKernel, v -> v * CNN.ALPHA),
                        v1 -> v2 -> v1 + v2);
                kernels[i][j] = deltaKernel;
            }
        }
	}

	@Override
	public void updateBias() {
		for (int j = 0; j < outCount; j++) {
            double[][] error = sum(errors, j);
            double deltaBias = sum(error) / batchSize;
            double bias = this.bias[j] + CNN.ALPHA * deltaBias;
            this.bias[j] = bias;
        }
	}
	
	public static int getMaxIndex(double[] out) {
        double max = out[0];
        int index = 0;
        for (int i = 1; i < out.length; i++)
            if (out[i] > max) {
                max = out[i];
                index = i;
            }
        return index;
    }
	
	/**
     * �Ծ���Ԫ�����
     *
     * @param error
     * @return ע�������ͺܿ��ܻ����
     */

    public static double sum(double[][] error) {
        int m = error.length;
        int n = error[0].length;
        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                sum += error[i][j];
            }
        }
        return sum;
    }
    
    /**
     * ��errors[...][j]Ԫ�����
     *
     * @param errors
     * @param j
     * @return
     */
    public static double[][] sum(double[][][][] errors, int j) {
        int m = errors[0][j].length;
        int n = errors[0][j][0].length;
        double[][] result = new double[m][n];
        for (int mi = 0; mi < m; mi++) {
            for (int nj = 0; nj < n; nj++) {
                double sum = 0;
                for (int i = 0; i < errors.length; i++)
                    sum += errors[i][j][mi][nj];
                result[mi][nj] = sum;
            }
        }
        return result;
    }
    
    private void setOutput(int recordIndex, DataSet.Record record, Layer lastLayer) {
        //TODO: ���Բ��л�
        for (int j = 0; j < outs[recordIndex].length; j++) {
            final int finalJ = j;
            final double[][][] sum = {null};// ��ÿһ������map�ľ���������
            lastLayer.forEachOutput(recordIndex, i -> lastRectangle -> {
                double[][] kernel = kernels[i][finalJ];
                if (sum[0] == null) {
                    sum[0] = MathUtil.validConvolutional(lastRectangle, kernel);
                } else {
                    sum[0] = MathUtil.trans(MathUtil.validConvolutional(lastRectangle, kernel), sum[0], v1 -> v2 -> v1 + v2);
                }
            });
            final double bias = this.bias[j];
            sum[0] = MathUtil.trans(sum[0], value -> MathUtil.sigmoid(value + bias));
            outs[recordIndex][j] = sum[0];
        }
    }
    
    public int getLabel(int recordIndex) {
        double[] realOuts = Arrays.stream(this.outs[recordIndex]).mapToDouble(out -> out[0][0]).toArray();
        return getMaxIndex(realOuts);
    }
    
    private boolean setErrors(int recordIndex, DataSet.Record record) {
        double[] target = new double[outCount];
        double[] realOuts = Arrays.stream(this.outs[recordIndex]).mapToDouble(out -> out[0][0]).toArray();
        int label = record.getLabel();
        target[label] = 1;
        for (int i = 0; i < outCount; i++) {
            errors[recordIndex][i][0][0] = realOuts[i] * (1 - realOuts[i]) * (target[i] - realOuts[i]);
        }
        return label == getMaxIndex(realOuts);
    }

}
