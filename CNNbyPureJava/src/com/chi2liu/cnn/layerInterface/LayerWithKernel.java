/**
 * @projectName: CNNbyPureJava
 * @packageName: com.chi2liu.cnn.layerInterface
 * @fileName: LayerWithKernel.java
 */
package com.chi2liu.cnn.layerInterface;

/**
 * @author: chi633 
 * @Email: cliu_whu@yeah.net
 * @date��2017��12��10�� ����4:00:38
 * @className: LayerWithKernel.java
 * @version: 1.0
 */
public interface LayerWithKernel extends Layer {
	
	/**
	 * ��ȡһ��kernel
	 * ���kernel�������Ǻ�ǰһ���ĳ�����preLayerOutIndex
	 * ���м����
	 * �õ������ĳ�����thisLayerOutIndex
	 * @param preLayerOutIndex
	 * @param thisLayerOutIndex
	 * @return
	 * @date: 2017��12��10��
	 */
	double[][] getKernel(int preLayerOutIndex, int thisLayerOutIndex);

    void updateKernels();

    void updateBias();

}
