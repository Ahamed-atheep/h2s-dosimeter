"""
AS7343 Multispectral Ratio Model Engine.
Direct Python implementation of the 50-tree Random Forest regressor
defined in firmware/hydrotrack_integrated_light/as7343_ratio_model.c.

Operates on 12 spectral channel ratios normalized by the Clear channel:
  input[0]  = F1_405nm  / Clear
  input[1]  = F2_425nm  / Clear
  input[2]  = FZ_450nm  / Clear
  input[3]  = F3_475nm  / Clear
  input[4]  = F4_515nm  / Clear
  input[5]  = F5_550nm  / Clear
  input[6]  = FY_555nm  / Clear
  input[7]  = FXL_600nm / Clear
  input[8]  = F6_640nm  / Clear
  input[9]  = F7_690nm  / Clear
  input[10] = F8_745nm  / Clear
  input[11] = NIR_855nm / Clear
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from src.utils.logger import get_logger

logger = get_logger("AS7343_Model")

# 12 AS7343 spectral channels in exact training/inference order
SPECTRAL_CHANNELS = [
    "F1", "F2", "FZ", "F3", "F4", "F5",
    "FY", "FXL", "F6", "F7", "F8", "NIR"
]

WAVELENGTH_MAP = {
    "F1": 405,
    "F2": 425,
    "FZ": 450,
    "F3": 475,
    "F4": 515,
    "F5": 550,
    "FY": 555,
    "FXL": 600,
    "F6": 640,
    "F7": 690,
    "F8": 745,
    "NIR": 855,
    "Clear": 550  # Broadband reference
}


def evaluate_raw_ratio_trees(input: List[float]) -> float:
    var0 = 0.0
    if input[7] <= 0.7854319512844086:
        if input[0] <= 0.02240685559809208:
            var0 = 5.0
        else:
            if input[11] <= 0.0190473310649395:
                var0 = 12.75
            else:
                var0 = 16.4
    else:
        if input[4] <= 0.5044668614864349:
            if input[10] <= 0.04329468868672848:
                var0 = 12.0
            else:
                var0 = 9.5
        else:
            if input[0] <= 0.02805284596979618:
                var0 = 5.714285714285714
            else:
                var0 = 8.071428571428571
    var1 = 0.0
    if input[8] <= 0.726056694984436:
        if input[11] <= 0.01957330573350191:
            if input[2] <= 0.38940465450286865:
                var1 = 13.285714285714286
            else:
                var1 = 11.666666666666666
        else:
            if input[3] <= 0.40818919241428375:
                var1 = 14.5
            else:
                var1 = 17.375
    else:
        if input[11] <= 0.03097666334360838:
            if input[4] <= 0.49851515889167786:
                var1 = 10.714285714285714
            else:
                var1 = 9.333333333333334
        else:
            if input[8] <= 0.7485114336013794:
                var1 = 7.0
            else:
                var1 = 5.333333333333333
    var2 = 0.0
    if input[7] <= 0.7823125123977661:
        if input[11] <= 0.01939673349261284:
            if input[2] <= 0.558188259601593:
                var2 = 13.333333333333334
            else:
                var2 = 5.0
        else:
            if input[11] <= 0.020633744075894356:
                var2 = 19.0
            else:
                var2 = 15.909090909090908
    else:
        if input[10] <= 0.04524254985153675:
            if input[6] <= 0.8395374119281769:
                var2 = 10.125
            else:
                var2 = 5.0
        else:
            if input[11] <= 0.033795639872550964:
                var2 = 7.571428571428571
            else:
                var2 = 5.8
    var3 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[6] <= 0.7388602197170258:
            if input[7] <= 0.7227454483509064:
                var3 = 13.2
            else:
                var3 = 11.8
        else:
            if input[7] <= 0.733611524105072:
                var3 = 19.0
            else:
                var3 = 15.7
    else:
        if input[4] <= 0.5086089372634888:
            if input[11] <= 0.027244159020483494:
                var3 = 12.0
            else:
                var3 = 9.1875
        else:
            if input[8] <= 0.6513493657112122:
                var3 = 11.0
            else:
                var3 = 6.153846153846154
    var4 = 0.0
    if input[11] <= 0.026547610759735107:
        if input[4] <= 0.5796748399734497:
            if input[3] <= 0.397264689207077:
                var4 = 13.222222222222221
            else:
                var4 = 16.272727272727273
        else:
            var4 = 5.0
    else:
        if input[4] <= 0.5044668614864349:
            if input[6] <= 0.7380950748920441:
                var4 = 11.0
            else:
                var4 = 9.666666666666666
        else:
            if input[6] <= 0.7562699317932129:
                var4 = 7.0
            else:
                var4 = 5.444444444444445
    var5 = 0.0
    if input[1] <= 0.0718407891690731:
        if input[10] <= 0.04516918584704399:
            if input[6] <= 0.7380950748920441:
                var5 = 11.0
            else:
                var5 = 9.6
        else:
            if input[10] <= 0.049317821860313416:
                var5 = 7.545454545454546
            else:
                var5 = 5.5
    else:
        if input[11] <= 0.01957330573350191:
            if input[9] <= 0.3008236736059189:
                var5 = 13.285714285714286
            else:
                var5 = 11.666666666666666
        else:
            if input[11] <= 0.02504850924015045:
                var5 = 17.0
            else:
                var5 = 13.333333333333334
    var6 = 0.0
    if input[7] <= 0.7871067821979523:
        if input[2] <= 0.5063674002885818:
            if input[11] <= 0.019625463522970676:
                var6 = 13.285714285714286
            else:
                var6 = 16.58823529411765
        else:
            var6 = 5.0
    else:
        if input[4] <= 0.5004181563854218:
            if input[11] <= 0.026877176016569138:
                var6 = 12.4
            else:
                var6 = 10.142857142857142
        else:
            if input[11] <= 0.03373376838862896:
                var6 = 7.615384615384615
            else:
                var6 = 5.428571428571429
    var7 = 0.0
    if input[11] <= 0.026547610759735107:
        if input[2] <= 0.558188259601593:
            if input[11] <= 0.019625463522970676:
                var7 = 12.9
            else:
                var7 = 16.476190476190474
        else:
            var7 = 5.0
    else:
        if input[4] <= 0.5088009238243103:
            if input[4] <= 0.49557994306087494:
                var7 = 11.4
            else:
                var7 = 9.555555555555555
        else:
            if input[0] <= 0.028355580754578114:
                var7 = 5.75
            else:
                var7 = 7.166666666666667
    var8 = 0.0
    if input[8] <= 0.7151843011379242:
        if input[3] <= 0.5446776449680328:
            if input[11] <= 0.020600914023816586:
                var8 = 19.0
            else:
                var8 = 15.866666666666667
        else:
            if input[5] <= 0.19799760729074478:
                var8 = 13.0
            else:
                var8 = 11.5
    else:
        if input[10] <= 0.04524254985153675:
            if input[5] <= 0.1847192645072937:
                var8 = 9.538461538461538
            else:
                var8 = 11.666666666666666
        else:
            if input[10] <= 0.04921344108879566:
                var8 = 7.555555555555555
            else:
                var8 = 5.166666666666667
    var9 = 0.0
    if input[8] <= 0.7231029868125916:
        if input[11] <= 0.01957330573350191:
            if input[9] <= 0.23728825896978378:
                var9 = 14.0
            else:
                var9 = 12.8
        else:
            if input[11] <= 0.02494814805686474:
                var9 = 17.823529411764707
            else:
                var9 = 14.555555555555555
    else:
        if input[10] <= 0.04524254985153675:
            if input[3] <= 0.347343847155571:
                var9 = 9.333333333333334
            else:
                var9 = 11.4
        else:
            if input[10] <= 0.04907453432679176:
                var9 = 7.5
            else:
                var9 = 5.75
    var10 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[1] <= 0.10924173519015312:
            if input[5] <= 0.19966813921928406:
                var10 = 16.954545454545453
            else:
                var10 = 13.333333333333334
        else:
            var10 = 5.0
    else:
        if input[11] <= 0.030898750759661198:
            if input[2] <= 0.39954017102718353:
                var10 = 10.529411764705882
            else:
                var10 = 5.0
        else:
            if input[11] <= 0.03371952846646309:
                var10 = 7.25
            else:
                var10 = 5.625
    var11 = 0.0
    if input[7] <= 0.7871067821979523:
        if input[1] <= 0.10756824538111687:
            if input[6] <= 0.7379933595657349:
                var11 = 13.25
            else:
                var11 = 16.869565217391305
        else:
            var11 = 5.0
    else:
        if input[4] <= 0.5004181563854218:
            if input[7] <= 0.7925025522708893:
                var11 = 13.0
            else:
                var11 = 10.285714285714286
        else:
            if input[0] <= 0.028100095689296722:
                var11 = 5.857142857142857
            else:
                var11 = 8.444444444444445
    var12 = 0.0
    if input[7] <= 0.7871067821979523:
        if input[4] <= 0.5690710246562958:
            if input[6] <= 0.7393021285533905:
                var12 = 13.625
            else:
                var12 = 16.25
        else:
            var12 = 5.0
    else:
        if input[4] <= 0.49851515889167786:
            if input[0] <= 0.028988643549382687:
                var12 = 11.0
            else:
                var12 = 12.166666666666666
        else:
            if input[11] <= 0.033795639872550964:
                var12 = 7.684210526315789
            else:
                var12 = 5.875
    var13 = 0.0
    if input[0] <= 0.029564332216978073:
        if input[4] <= 0.5041150748729706:
            if input[10] <= 0.04030942730605602:
                var13 = 14.0
            else:
                var13 = 11.0
        else:
            if input[7] <= 0.7137116491794586:
                var13 = 19.0
            else:
                var13 = 6.909090909090909
    else:
        if input[7] <= 0.7854319512844086:
            if input[11] <= 0.01876678690314293:
                var13 = 12.5
            else:
                var13 = 17.692307692307693
        else:
            if input[1] <= 0.07644764333963394:
                var13 = 12.0
            else:
                var13 = 11.0
    var14 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[9] <= 0.21090150624513626:
            var14 = 5.0
        else:
            if input[6] <= 0.741215705871582:
                var14 = 13.777777777777779
            else:
                var14 = 16.789473684210527
    else:
        if input[10] <= 0.04551668278872967:
            if input[0] <= 0.024543683975934982:
                var14 = 5.0
            else:
                var14 = 10.2
        else:
            if input[10] <= 0.049317821860313416:
                var14 = 7.428571428571429
            else:
                var14 = 5.75
    var15 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[11] <= 0.01920331548899412:
            if input[1] <= 0.0783938355743885:
                var15 = 14.0
            else:
                var15 = 12.88888888888889
        else:
            if input[8] <= 0.5916441679000854:
                var15 = 19.0
            else:
                var15 = 15.636363636363637
    else:
        if input[10] <= 0.04524254985153675:
            if input[7] <= 0.8270289897918701:
                var15 = 10.5
            else:
                var15 = 5.0
        else:
            if input[11] <= 0.03373376838862896:
                var15 = 7.75
            else:
                var15 = 5.833333333333333
    var16 = 0.0
    if input[11] <= 0.026877176016569138:
        if input[11] <= 0.019625463522970676:
            if input[8] <= 0.5793278813362122:
                var16 = 13.428571428571429
            else:
                var16 = 11.666666666666666
        else:
            if input[3] <= 0.407839372754097:
                var16 = 13.916666666666666
            else:
                var16 = 17.5
    else:
        if input[10] <= 0.04524254985153675:
            if input[11] <= 0.02843502350151539:
                var16 = 10.285714285714286
            else:
                var16 = 9.0
        else:
            if input[4] <= 0.5161913335323334:
                var16 = 8.0
            else:
                var16 = 6.833333333333333
    var17 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[11] <= 0.019625463522970676:
            if input[8] <= 0.5793278813362122:
                var17 = 13.0
            else:
                var17 = 12.0
        else:
            if input[10] <= 0.031902176328003407:
                var17 = 19.0
            else:
                var17 = 15.555555555555555
    else:
        if input[4] <= 0.5071076452732086:
            if input[11] <= 0.027244159020483494:
                var17 = 11.75
            else:
                var17 = 9.6
        else:
            if input[0] <= 0.03228670172393322:
                var17 = 6.666666666666667
            else:
                var17 = 11.0
    var18 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[11] <= 0.019344575703144073:
            if input[11] <= 0.01879806909710169:
                var18 = 12.777777777777779
            else:
                var18 = 5.0
        else:
            if input[11] <= 0.020600914023816586:
                var18 = 19.0
            else:
                var18 = 16.0
    else:
        if input[11] <= 0.03175953961908817:
            if input[4] <= 0.4957309514284134:
                var18 = 12.0
            else:
                var18 = 8.785714285714286
        else:
            if input[8] <= 0.7426301836967468:
                var18 = 7.0
            else:
                var18 = 5.333333333333333
    var19 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[2] <= 0.558188259601593:
            if input[6] <= 0.741566002368927:
                var19 = 13.5
            else:
                var19 = 16.842105263157894
        else:
            var19 = 5.0
    else:
        if input[4] <= 0.5071076452732086:
            if input[3] <= 0.34911464154720306:
                var19 = 9.9
            else:
                var19 = 11.75
        else:
            if input[0] <= 0.032187859527766705:
                var19 = 6.454545454545454
            else:
                var19 = 11.0
    var20 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[11] <= 0.019625463522970676:
            if input[1] <= 0.0783938355743885:
                var20 = 14.0
            else:
                var20 = 12.75
        else:
            if input[8] <= 0.5906540751457214:
                var20 = 19.0
            else:
                var20 = 15.928571428571429
    else:
        if input[4] <= 0.5044668614864349:
            if input[0] <= 0.0287614818662405:
                var20 = 9.5
            else:
                var20 = 11.8
        else:
            if input[0] <= 0.03265125025063753:
                var20 = 6.428571428571429
            else:
                var20 = 11.0
    var21 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[2] <= 0.4930834472179413:
            if input[11] <= 0.01957330573350191:
                var21 = 13.0
            else:
                var21 = 17.0625
        else:
            var21 = 5.0
    else:
        if input[10] <= 0.044867003336548805:
            if input[7] <= 0.8270289897918701:
                var21 = 10.846153846153847
            else:
                var21 = 5.0
        else:
            if input[8] <= 0.7527126371860504:
                var21 = 7.285714285714286
            else:
                var21 = 5.625
    var22 = 0.0
    if input[7] <= 0.789028137922287:
        if input[1] <= 0.08853249996900558:
            if input[0] <= 0.027931579388678074:
                var22 = 13.714285714285714
            else:
                var22 = 15.666666666666666
        else:
            var22 = 19.0
    else:
        if input[4] <= 0.5020971149206161:
            if input[7] <= 0.8062112033367157:
                var22 = 9.833333333333334
            else:
                var22 = 11.5
        else:
            if input[6] <= 0.7463415563106537:
                var22 = 8.714285714285714
            else:
                var22 = 6.466666666666667
    var23 = 0.0
    if input[7] <= 0.7910954058170319:
        if input[4] <= 0.5796748399734497:
            if input[7] <= 0.69828200340271:
                var23 = 19.0
            else:
                var23 = 14.689655172413794
        else:
            var23 = 5.0
    else:
        if input[4] <= 0.5044668614864349:
            if input[10] <= 0.04351297952234745:
                var23 = 11.75
            else:
                var23 = 8.75
        else:
            if input[0] <= 0.02769437897950411:
                var23 = 5.5
            else:
                var23 = 7.666666666666667
    var24 = 0.0
    if input[8] <= 0.7255269289016724:
        if input[7] <= 0.7866645753383636:
            if input[11] <= 0.01957330573350191:
                var24 = 13.0
            else:
                var24 = 16.42105263157895
        else:
            if input[4] <= 0.4992096275091171:
                var24 = 13.0
            else:
                var24 = 11.333333333333334
    else:
        if input[11] <= 0.03183565102517605:
            if input[4] <= 0.49851515889167786:
                var24 = 11.2
            else:
                var24 = 8.6
        else:
            if input[10] <= 0.05146055109798908:
                var24 = 6.4
            else:
                var24 = 5.0
    var25 = 0.0
    if input[8] <= 0.7284266948699951:
        if input[2] <= 0.558188259601593:
            if input[8] <= 0.7141333520412445:
                var25 = 15.307692307692308
            else:
                var25 = 12.222222222222221
        else:
            var25 = 5.0
    else:
        if input[11] <= 0.03149895370006561:
            if input[5] <= 0.18530020862817764:
                var25 = 9.285714285714286
            else:
                var25 = 12.0
        else:
            if input[5] <= 0.18507494032382965:
                var25 = 5.75
            else:
                var25 = 7.0
    var26 = 0.0
    if input[7] <= 0.7910954058170319:
        if input[11] <= 0.01939673349261284:
            if input[2] <= 0.48270153999328613:
                var26 = 11.0
            else:
                var26 = 5.0
        else:
            if input[10] <= 0.0382865946739912:
                var26 = 18.333333333333332
            else:
                var26 = 15.0
    else:
        if input[4] <= 0.5044668614864349:
            if input[11] <= 0.027244159020483494:
                var26 = 11.75
            else:
                var26 = 9.8
        else:
            if input[6] <= 0.7562699317932129:
                var26 = 7.421052631578948
            else:
                var26 = 5.25
    var27 = 0.0
    if input[7] <= 0.7835066318511963:
        if input[2] <= 0.558188259601593:
            if input[6] <= 0.7388602197170258:
                var27 = 13.0
            else:
                var27 = 16.65
        else:
            var27 = 5.0
    else:
        if input[11] <= 0.031717490404844284:
            if input[0] <= 0.02823895774781704:
                var27 = 8.166666666666666
            else:
                var27 = 11.285714285714286
        else:
            if input[10] <= 0.049317821860313416:
                var27 = 7.5
            else:
                var27 = 5.7
    var28 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[11] <= 0.01957330573350191:
            if input[5] <= 0.16355183720588684:
                var28 = 14.0
            else:
                var28 = 12.75
        else:
            if input[11] <= 0.020633744075894356:
                var28 = 19.0
            else:
                var28 = 16.0
    else:
        if input[4] <= 0.5033652186393738:
            if input[10] <= 0.04216683842241764:
                var28 = 12.333333333333334
            else:
                var28 = 10.571428571428571
        else:
            if input[0] <= 0.02805284596979618:
                var28 = 6.0
            else:
                var28 = 8.714285714285714
    var29 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[2] <= 0.3254148066043854:
            if input[8] <= 0.7165033519268036:
                var29 = 14.166666666666666
            else:
                var29 = 11.0
        else:
            if input[11] <= 0.019625463522970676:
                var29 = 13.4
            else:
                var29 = 16.666666666666668
    else:
        if input[4] <= 0.49851515889167786:
            if input[9] <= 0.39125752449035645:
                var29 = 12.0
            else:
                var29 = 10.666666666666666
        else:
            if input[0] <= 0.032287683337926865:
                var29 = 7.235294117647059
            else:
                var29 = 11.0
    var30 = 0.0
    if input[8] <= 0.7254729866981506:
        if input[1] <= 0.10746371001005173:
            if input[2] <= 0.3140113800764084:
                var30 = 12.8
            else:
                var30 = 15.615384615384615
        else:
            var30 = 5.0
    else:
        if input[10] <= 0.04516918584704399:
            if input[4] <= 0.49557994306087494:
                var30 = 11.222222222222221
            else:
                var30 = 9.75
        else:
            if input[10] <= 0.049456728622317314:
                var30 = 7.714285714285714
            else:
                var30 = 5.75
    var31 = 0.0
    if input[7] <= 0.7823125123977661:
        if input[10] <= 0.03422888554632664:
            if input[2] <= 0.4866851717233658:
                var31 = 13.25
            else:
                var31 = 5.0
        else:
            if input[3] <= 0.407839372754097:
                var31 = 15.0
            else:
                var31 = 17.153846153846153
    else:
        if input[10] <= 0.04516918584704399:
            if input[10] <= 0.04216683842241764:
                var31 = 11.333333333333334
            else:
                var31 = 10.0
        else:
            if input[10] <= 0.04964105226099491:
                var31 = 7.454545454545454
            else:
                var31 = 5.0
    var32 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[11] <= 0.019625463522970676:
            if input[1] <= 0.06482306309044361:
                var32 = 14.0
            else:
                var32 = 12.666666666666666
        else:
            if input[1] <= 0.07518237829208374:
                var32 = 14.6
            else:
                var32 = 17.25
    else:
        if input[4] <= 0.5071076452732086:
            if input[11] <= 0.027244159020483494:
                var32 = 11.8
            else:
                var32 = 9.416666666666666
        else:
            if input[2] <= 0.39954017102718353:
                var32 = 7.5
            else:
                var32 = 5.0
    var33 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[1] <= 0.10756824538111687:
            if input[1] <= 0.08853249996900558:
                var33 = 14.476190476190476
            else:
                var33 = 19.0
        else:
            var33 = 5.0
    else:
        if input[10] <= 0.0450496356934309:
            if input[2] <= 0.39954017102718353:
                var33 = 10.3125
            else:
                var33 = 5.0
        else:
            if input[11] <= 0.03373376838862896:
                var33 = 7.5
            else:
                var33 = 5.7
    var34 = 0.0
    if input[8] <= 0.7255269289016724:
        if input[5] <= 0.21120809018611908:
            if input[7] <= 0.7866645753383636:
                var34 = 15.36
            else:
                var34 = 11.88888888888889
        else:
            var34 = 5.0
    else:
        if input[10] <= 0.04524254985153675:
            if input[4] <= 0.49851515889167786:
                var34 = 11.25
            else:
                var34 = 9.75
        else:
            if input[8] <= 0.763880580663681:
                var34 = 8.0
            else:
                var34 = 5.8
    var35 = 0.0
    if input[11] <= 0.026547610759735107:
        if input[5] <= 0.21120809018611908:
            if input[11] <= 0.019151157699525356:
                var35 = 13.1
            else:
                var35 = 16.0
        else:
            var35 = 5.0
    else:
        if input[4] <= 0.5102689862251282:
            if input[4] <= 0.49557994306087494:
                var35 = 11.333333333333334
            else:
                var35 = 9.0
        else:
            if input[6] <= 0.8168790638446808:
                var35 = 7.0
            else:
                var35 = 5.0
    var36 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[11] <= 0.01957330573350191:
            if input[5] <= 0.16486424952745438:
                var36 = 14.0
            else:
                var36 = 12.666666666666666
        else:
            if input[9] <= 0.324161559343338:
                var36 = 19.0
            else:
                var36 = 15.909090909090908
    else:
        if input[11] <= 0.030680214054882526:
            if input[9] <= 0.33644524216651917:
                var36 = 5.0
            else:
                var36 = 10.5
        else:
            if input[10] <= 0.049554597586393356:
                var36 = 7.666666666666667
            else:
                var36 = 5.777777777777778
    var37 = 0.0
    if input[11] <= 0.02863046620041132:
        if input[0] <= 0.022337211295962334:
            var37 = 5.0
        else:
            if input[7] <= 0.7866645753383636:
                var37 = 15.130434782608695
            else:
                var37 = 11.666666666666666
    else:
        if input[11] <= 0.031717490404844284:
            if input[3] <= 0.3169267624616623:
                var37 = 11.0
            else:
                var37 = 8.625
        else:
            if input[8] <= 0.7468864619731903:
                var37 = 7.0
            else:
                var37 = 5.8
    var38 = 0.0
    if input[7] <= 0.7823125123977661:
        if input[11] <= 0.01957330573350191:
            if input[1] <= 0.10924173519015312:
                var38 = 12.8
            else:
                var38 = 5.0
        else:
            if input[11] <= 0.021192104555666447:
                var38 = 19.0
            else:
                var38 = 16.333333333333332
    else:
        if input[4] <= 0.5041150748729706:
            if input[10] <= 0.044916411861777306:
                var38 = 11.25
            else:
                var38 = 8.4
        else:
            if input[0] <= 0.028026522137224674:
                var38 = 5.714285714285714
            else:
                var38 = 7.1
    var39 = 0.0
    if input[7] <= 0.7871067821979523:
        if input[9] <= 0.21090150624513626:
            var39 = 5.0
        else:
            if input[6] <= 0.7388602197170258:
                var39 = 13.75
            else:
                var39 = 17.333333333333332
    else:
        if input[4] <= 0.5044668614864349:
            if input[11] <= 0.027244159020483494:
                var39 = 11.666666666666666
            else:
                var39 = 9.444444444444445
        else:
            if input[4] <= 0.514904797077179:
                var39 = 8.25
            else:
                var39 = 6.111111111111111
    var40 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[11] <= 0.01957330573350191:
            if input[4] <= 0.4411756098270416:
                var40 = 14.0
            else:
                var40 = 12.666666666666666
        else:
            if input[9] <= 0.324161559343338:
                var40 = 19.0
            else:
                var40 = 16.125
    else:
        if input[10] <= 0.04548989608883858:
            if input[6] <= 0.8395374119281769:
                var40 = 10.384615384615385
            else:
                var40 = 5.0
        else:
            if input[11] <= 0.033795639872550964:
                var40 = 7.571428571428571
            else:
                var40 = 5.8
    var41 = 0.0
    if input[8] <= 0.7254729866981506:
        if input[1] <= 0.10924173519015312:
            if input[11] <= 0.019625463522970676:
                var41 = 12.555555555555555
            else:
                var41 = 16.166666666666668
        else:
            var41 = 5.0
    else:
        if input[10] <= 0.04524254985153675:
            if input[4] <= 0.49851515889167786:
                var41 = 11.2
            else:
                var41 = 9.333333333333334
        else:
            if input[10] <= 0.049317821860313416:
                var41 = 7.666666666666667
            else:
                var41 = 5.75
    var42 = 0.0
    if input[8] <= 0.726056694984436:
        if input[11] <= 0.01957330573350191:
            if input[9] <= 0.3008236736059189:
                var42 = 13.428571428571429
            else:
                var42 = 11.5
        else:
            if input[11] <= 0.020633744075894356:
                var42 = 19.0
            else:
                var42 = 15.25
    else:
        if input[11] <= 0.030154738575220108:
            if input[1] <= 0.06871531903743744:
                var42 = 9.428571428571429
            else:
                var42 = 12.0
        else:
            if input[10] <= 0.049317821860313416:
                var42 = 7.454545454545454
            else:
                var42 = 5.6
    var43 = 0.0
    if input[11] <= 0.026547610759735107:
        if input[2] <= 0.3142838627099991:
            if input[11] <= 0.02622964046895504:
                var43 = 13.0
            else:
                var43 = 12.0
        else:
            if input[11] <= 0.01957330573350191:
                var43 = 12.666666666666666
            else:
                var43 = 16.428571428571427
    else:
        if input[4] <= 0.5117031931877136:
            if input[10] <= 0.044916411861777306:
                var43 = 10.0
            else:
                var43 = 8.2
        else:
            if input[7] <= 0.8211188316345215:
                var43 = 6.555555555555555
            else:
                var43 = 5.0
    var44 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[2] <= 0.558188259601593:
            if input[1] <= 0.07418910413980484:
                var44 = 13.428571428571429
            else:
                var44 = 17.53846153846154
        else:
            var44 = 5.0
    else:
        if input[11] <= 0.029734043404459953:
            if input[11] <= 0.02635166049003601:
                var44 = 12.0
            else:
                var44 = 9.23076923076923
        else:
            if input[11] <= 0.03393353335559368:
                var44 = 7.666666666666667
            else:
                var44 = 5.75
    var45 = 0.0
    if input[7] <= 0.7866645753383636:
        if input[11] <= 0.01957330573350191:
            if input[1] <= 0.10924173519015312:
                var45 = 13.166666666666666
            else:
                var45 = 5.0
        else:
            if input[9] <= 0.324161559343338:
                var45 = 19.0
            else:
                var45 = 16.363636363636363
    else:
        if input[10] <= 0.044916411861777306:
            if input[3] <= 0.35645677149295807:
                var45 = 10.125
            else:
                var45 = 12.0
        else:
            if input[11] <= 0.03175953961908817:
                var45 = 8.285714285714286
            else:
                var45 = 6.7
    var46 = 0.0
    if input[7] <= 0.7871067821979523:
        if input[11] <= 0.01920331548899412:
            if input[4] <= 0.4520644098520279:
                var46 = 14.0
            else:
                var46 = 13.0
        else:
            if input[4] <= 0.53437140583992:
                var46 = 17.181818181818183
            else:
                var46 = 14.666666666666666
    else:
        if input[4] <= 0.514919251203537:
            if input[1] <= 0.06902769207954407:
                var46 = 8.714285714285714
            else:
                var46 = 12.5
        else:
            if input[2] <= 0.3395157754421234:
                var46 = 6.166666666666667
            else:
                var46 = 5.0
    var47 = 0.0
    if input[8] <= 0.7255269289016724:
        if input[5] <= 0.21120809018611908:
            if input[0] <= 0.029485631734132767:
                var47 = 14.466666666666667
            else:
                var47 = 17.0
        else:
            var47 = 5.0
    else:
        if input[4] <= 0.5071076452732086:
            if input[7] <= 0.8048708736896515:
                var47 = 9.416666666666666
            else:
                var47 = 11.333333333333334
        else:
            if input[5] <= 0.2312314137816429:
                var47 = 7.375
            else:
                var47 = 5.0
    var48 = 0.0
    if input[11] <= 0.026547610759735107:
        if input[5] <= 0.21120809018611908:
            if input[3] <= 0.40427546203136444:
                var48 = 13.125
            else:
                var48 = 16.695652173913043
        else:
            var48 = 5.0
    else:
        if input[10] <= 0.04534653201699257:
            if input[1] <= 0.07997797057032585:
                var48 = 11.0
            else:
                var48 = 5.0
        else:
            if input[11] <= 0.033566344529390335:
                var48 = 7.75
            else:
                var48 = 5.818181818181818
    var49 = 0.0
    if input[3] <= 0.35645677149295807:
        if input[11] <= 0.029734043404459953:
            if input[6] <= 0.7380950748920441:
                var49 = 11.0
            else:
                var49 = 9.714285714285714
        else:
            if input[10] <= 0.049317821860313416:
                var49 = 7.5
            else:
                var49 = 6.0
    else:
        if input[11] <= 0.019344575703144073:
            if input[9] <= 0.23660285770893097:
                var49 = 5.0
            else:
                var49 = 12.4
        else:
            if input[2] <= 0.3140113800764084:
                var49 = 12.25
            else:
                var49 = 16.8
    return (var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + var11 + var12 + var13 + var14 + var15 + var16 + var17 + var18 + var19 + var20 + var21 + var22 + var23 + var24 + var25 + var26 + var27 + var28 + var29 + var30 + var31 + var32 + var33 + var34 + var35 + var36 + var37 + var38 + var39 + var40 + var41 + var42 + var43 + var44 + var45 + var46 + var47 + var48 + var49) * 0.02


def score_as7343_ratio(channels_over_clear: List[float]) -> float:
    """
    Compute cumulative dose estimate (ppm*h) from 12 normalized AS7343 ratios.
    """
    if len(channels_over_clear) != 12:
        raise ValueError(f"Expected 12 spectral ratios, got {len(channels_over_clear)}")
    return evaluate_raw_ratio_trees(channels_over_clear)


@dataclass
class AS7343PredictionResult:
    cumulative_dose_ppm_h: float
    twa_8h_equivalent_ppm: float
    risk_level: str
    risk_tier: str
    risk_color: str
    action_recommendation: str
    ratios: Dict[str, float]
    raw_channels: Dict[str, float]
    clear_count: float
    is_saturated: bool
    model_type: str = "RANDOM_FOREST_AS7343"
    model_version: str = "v1.0-ratio-50trees"


class AS7343Predictor:
    """
    High-level AS7343 Multispectral Dosimeter Prediction and Risk Engine.
    """

    @staticmethod
    def classify_risk(dose_ppm_h: float) -> Tuple[str, str, str, str]:
        """
        Maps cumulative dose to standard occupational safety tiers.
        """
        if dose_ppm_h < 10.0:
            return (
                "SAFE / NORMAL",
                "SAFE",
                "#10b981",
                "Normal operations: Cumulative exposure is within permissible safe limits."
            )
        elif dose_ppm_h < 40.0:
            return (
                "WARNING / ELEVATED",
                "WARNING",
                "#f59e0b",
                "Caution: Elevated H2S exposure. Inspect local ventilation and verify badge seal."
            )
        elif dose_ppm_h < 80.0:
            return (
                "DANGER / ACTION REQUIRED",
                "DANGER",
                "#f97316",
                "Action Required: Exposure approaching ceiling limit. Rotate worker to fresh air zone."
            )
        else:
            return (
                "CRITICAL / THRESHOLD EXCEEDED",
                "CRITICAL",
                "#ef4444",
                "CRITICAL ALERT: Statutory exposure threshold exceeded! Evacuate area immediately and notify Safety Officer."
            )

    @classmethod
    def predict(
        cls,
        channels: Union[List[float], Dict[str, float]],
        clear: Optional[float] = None,
        exposure_duration_hours: float = 8.0,
        is_saturated: bool = False
    ) -> AS7343PredictionResult:
        """
        Predict cumulative dose from AS7343 channel counts.
        """
        raw_dict: Dict[str, float] = {}
        clear_val = 0.0

        if isinstance(channels, dict):
            raw_dict = {k: float(v) for k, v in channels.items()}
            clear_val = float(clear if clear is not None else raw_dict.get("Clear", raw_dict.get("CLEAR", 0.0)))
            ratio_vec = []
            for k in SPECTRAL_CHANNELS:
                if k not in raw_dict:
                    raise ValueError(f"Missing required AS7343 channel '{k}' in channel map.")
                if clear_val <= 0.0:
                    raise ValueError("Clear channel count must be strictly positive (> 0) for ratio normalization.")
                ratio_vec.append(raw_dict[k] / clear_val)
        elif isinstance(channels, (list, tuple)):
            if len(channels) == 12:
                if clear is None or clear <= 0.0:
                    raise ValueError("Must provide positive Clear count when passing 12 raw channels.")
                clear_val = float(clear)
                ratio_vec = [float(ch) / clear_val for ch in channels]
                raw_dict = {k: float(ch) for k, ch in zip(SPECTRAL_CHANNELS, channels)}
                raw_dict["Clear"] = clear_val
            elif len(channels) >= 13:
                # 14 channels standard (first 12 = F1..NIR, index 12 = Clear)
                clear_val = float(channels[12])
                if clear_val <= 0.0:
                    raise ValueError("Clear channel count at index 12 must be > 0.")
                ratio_vec = [float(channels[i]) / clear_val for i in range(12)]
                raw_dict = {k: float(channels[i]) for i, k in enumerate(SPECTRAL_CHANNELS)}
                raw_dict["Clear"] = clear_val
            else:
                raise ValueError(f"Expected at least 12 channels, got {len(channels)}")
        else:
            raise TypeError("channels must be a list or dict of spectral channel counts.")

        # Run 50-tree inference
        dose_ppm_h = score_as7343_ratio(ratio_vec)
        # Bounded between 0.0 and 150.0 ppm*h
        dose_ppm_h = max(0.0, float(dose_ppm_h))

        # 8-hr equivalent TWA calculation
        eff_hours = max(0.1, float(exposure_duration_hours))
        twa_ppm = round(dose_ppm_h / eff_hours, 2)

        risk_level, risk_tier, risk_color, action_rec = cls.classify_risk(dose_ppm_h)

        ratio_dict = {k: round(r, 6) for k, r in zip(SPECTRAL_CHANNELS, ratio_vec)}

        return AS7343PredictionResult(
            cumulative_dose_ppm_h=round(dose_ppm_h, 2),
            twa_8h_equivalent_ppm=twa_ppm,
            risk_level=risk_level,
            risk_tier=risk_tier,
            risk_color=risk_color,
            action_recommendation=action_rec,
            ratios=ratio_dict,
            raw_channels=raw_dict,
            clear_count=clear_val,
            is_saturated=is_saturated
        )
