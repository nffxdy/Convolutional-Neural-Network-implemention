/**
 * @projectName: CNNbyPureJava
 * @packageName: com.chi2liu.cnn.layerImplement
 * @fileName: SampleLayer.java
 */
package com.chi2liu.cnn.layerImplement;

import java.util.function.Consumer;
import java.util.function.Function;

import com.chi2liu.cnn.data.DataSet;
import com.chi2liu.cnn.data.DataSet.Record;
import com.chi2liu.cnn.layerInterface.Layer;
import com.chi2liu.cnn.layerInterface.LayerWithKernel;
import com.chi2liu.cnn.util.MathUtil;
import com.chi2liu.cnn.util.Size;


/**
 * @author: chi633 
 * @Email: cliu_whu@yeah.net
 * @date：2017年12月10日 下午4:19:08
 * @className: SampleLayer.java
 * @version: 1.0
 */
/**
 * @author: chi633 
 * @Email: cliu_whu@yeah.net
 * @date：2017年12月10日 下午4:32:12
 * @className: SampleLayer.java
 * @version: 1.0
 */
/**
 * @author: chi633 
 * @Email: cliu_whu@yeah.net
 * @date：2017年12月10日 下午4:34:30
 * @className: SampleLayer.java
 * @version: 1.0
 */
public class SampleLayer implements Layer {
	
	/**
	 * 采样层的scale的size大小
	 */
	private final Size scaleSize;
	
    /**
     * 输出map的个数
     */
    private final int outCount;
    
    /**
     * 采样层的前一层，因为采样层的outCount和前一层是一致的
     */
    private final Layer preLayer;
    
    /**
     * 采样层的输出map的Size
     */
    private final Size size;
    
    /**
     * 采样层的输出
     * 四维：
     * 1：recordIndex
     * 2:outIndex
     * 3,4:输出的map
     */
    private final double[][][][] outs;
    
    /**
     * 采样层的errors
     */
    private final double[][][][] errors;
    
    /**
     * 采样层的下一层
     */
    private Layer nextLayer;
    
    public SampleLayer(int batchSize, Size scaleSize, Layer preLayer) {
        this.scaleSize = scaleSize;
        this.outCount = preLayer.getOutCount();//采样层输出和上一层输出个数一致
        this.preLayer = preLayer;
        //采样层经过和scale池化之后输出map的size
        this.size = preLayer.getSize().divide(scaleSize);
        errors = new double[batchSize][outCount][size.getWidth()][size.getWidth()];
        outs = new double[batchSize][outCount][size.getWidth()][size.getHeight()];
    }

	@Override
	public Size getSize() {
		return size;
	}
	
	public Size getScaleSize() {
		return scaleSize;
	}

	@Override
	public int getOutCount() {
		return outCount;
	}

	@Override
	public void forEachOutput(int recordIndex, Function<Integer, Consumer<double[][]>> function) {
		for(int i=0;i<outCount;i++)
			function.apply(i).accept(outs[recordIndex][i]);
	}

	@Override
	public void setNextLayer(Layer layer) {
		this.nextLayer = layer;
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
	
	/**
	 * 实现采样层核心操作采样（池化）的函数
	 * @param matrix
	 * @param scale
	 * @return
	 * @date: 2017年12月10日
	 */
	public static double[][] scaleMatrix(final double[][] matrix,
            final Size scale) {
		
		int m = matrix.length;
        int n = matrix[0].length;
        final int sm = m / scale.getWidth();
        final int sn = n / scale.getHeight();
        final double[][] outMatrix = new double[sm][sn];
        if (sm * scale.getWidth() != m || sn * scale.getHeight() != n)
            throw new RuntimeException("scale不能整除matrix");
        final int size = scale.getWidth() * scale.getHeight();
        for (int i = 0; i < sm; i++) {
            for (int j = 0; j < sn; j++) {
                double sum = 0.0;
                for (int si = i * scale.getWidth(); si < (i + 1) * scale.getWidth(); si++) {
                    for (int sj = j * scale.getHeight(); sj < (j + 1) * scale.getHeight(); sj++) {
                        sum += matrix[si][sj];
                    }
                }
                outMatrix[i][j] = sum / size;
            }
        }
        return outMatrix;
		
	}
	
	/**
	 * 计算采样层的输出
	 * @param recordIndex
	 * @param record
	 * @param lastLayer
	 * @date: 2017年12月10日
	 */
	private void setOutput(int recordIndex, DataSet.Record record, Layer lastLayer) {
        lastLayer.forEachOutput(recordIndex, i -> lastRectangle -> {
            outs[recordIndex][i] = scaleMatrix(lastRectangle, scaleSize);
        });
    }
	
	/**
	 * 设置采样层的errors
	 * @param recordIndex
	 * @param record
	 * @return
	 * @date: 2017年12月10日
	 */
	private boolean setErrors(int recordIndex, DataSet.Record record) {
        if (nextLayer instanceof LayerWithKernel) {
            forEachOutput(recordIndex, i -> out -> {
                final double[][][] sum = {null};// 对每一个卷积进行求和
                nextLayer.forEachOutput(recordIndex, j -> nextOut -> {
                    double[][] nextError = nextLayer.getError(recordIndex, j);
                    double[][] nextKernel = ((LayerWithKernel) nextLayer).getKernel(i, j);
                    if (sum[0] == null)
                        sum[0] = MathUtil.fullConvolutional(nextError, MathUtil.transRotate180(nextKernel));
                    else
                        sum[0] = MathUtil.trans(MathUtil.fullConvolutional(nextError, MathUtil.transRotate180(nextKernel)),
                                sum[0],
                                v1 -> v2 -> v1 + v2);
                });
                errors[recordIndex][i] = sum[0];
            });
        }
        return true;
    }

}
