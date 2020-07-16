# 中華民國第 60 屆中小學科學展覽會 風起「電」湧 - 人工智慧於風力發電效率探討 主程式
#簡介
	本研究的目的是探討：運用人工智慧輔助設計風力發電機的葉片，以提升發電的效率。我們從二維葉片剖面出發，以NACA5410翼型為原型，利用XFoil翼型設計軟體與人工智慧找尋最佳升力與最佳升阻比之剖面形狀，經由風洞實驗得到理論值與本文實驗具有相似的趨勢；接著我們基於該剖面形狀用3D列印製作了長方翼，實際裝上發電機後也證實，最佳化後的葉片顯著比原型表現更好；我們接著把實驗擴展到三維，即漸縮翼與翼尖小翼，透過實驗證實這些技巧對於風電效率有顯著的助益；而除了幾何形狀外，因配重可儲存動能的特性，我們也探討葉片設計的結果對配重的影響；最後，由於最佳升阻比之葉片有較低的能量損耗，排列風電機組的組合也得到更好的效果。
  
#程式使用方式
1. 請使用Anaconda 3，以確保流程無誤。
2. 安裝 pytorch、cuda、xfoil、gym
3. 將 OpenAI Envs內的兩個py檔案複製到複製到 anaconda3\Lib\site-packages\gym
4. 將 OpenAI Envs/classic_control 複製到 anaconda3\Lib\site-packages\gym\envs\classic_control
5. 將 xfoil airfoil generator 內的generator資料夾複製到 anaconda3\Lib\site-packages\xfoil
6. 執行 Main code/Main code.py
