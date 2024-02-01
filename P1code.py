import numpy as np
import matplotlib.pyplot as plt

# code S1, S2
# S1 = 0b001
# S2 = 0b0110

def p1code_generation(S1, S2):
    ##### 9.8.2.1 Carrier Distribution in P1 symbol
    active_carrier_S1_1 = [44, 45, 47, 51, 54, 59, 62, 64, 65, 66, 70, 75, 78, 80, 81,
                        82, 84, 85, 87, 88, 89, 90, 94, 96, 97, 98, 102, 107, 110,
                        112, 113, 114, 116, 117, 119, 120, 121, 122, 124, 125, 127,
                        131, 132, 133, 135, 136, 137, 138, 142, 144, 145, 146, 148,
                        149, 151, 152, 153, 154, 158, 160, 161, 162, 166, 171]
    active_carrier_S2   = [172, 173, 175, 179, 182, 187, 190, 192, 193, 194, 198, 203, 206, 208, 209, 210,
                        212, 213, 215, 216, 217, 218, 222, 224, 225, 226, 230, 235, 238, 240, 241, 242,
                        244, 245, 247, 248, 249, 250, 252, 253, 255, 259, 260, 261, 263, 264, 265, 266,
                        270, 272, 273, 274, 276, 277, 279, 280, 281, 282, 286, 288, 289, 290, 294, 299,
                        300, 301, 303, 307, 310, 315, 318, 320, 321, 322, 326, 331, 334, 336, 337, 338,
                        340, 341, 343, 344, 345, 346, 350, 352, 353, 354, 358, 363, 364, 365, 367, 371,
                        374, 379, 382, 384, 385, 386, 390, 395, 396, 397, 399, 403, 406, 411, 412, 413,
                        415, 419, 420, 421, 423, 424, 425, 426, 428, 429, 431, 435, 438, 443, 446, 448,
                        449, 450, 454, 459, 462, 464, 465, 466, 468, 469, 471, 472, 473, 474, 478, 480,
                        481, 482, 486, 491, 494, 496, 497, 498, 500, 501, 503, 504, 505, 506, 508, 509,
                        511, 515, 516, 517, 519, 520, 521, 522, 526, 528, 529, 530, 532, 533, 535, 536,
                        537, 538, 542, 544, 545, 546, 550, 555, 558, 560, 561, 562, 564, 565, 567, 568,
                        569, 570, 572, 573, 575, 579, 580, 581, 583, 584, 585, 586, 588, 589, 591, 595,
                        598, 603, 604, 605, 607, 611, 612, 613, 615, 616, 617, 618, 622, 624, 625, 626,
                        628, 629, 631, 632, 633, 634, 636, 637, 639, 643, 644, 645, 647, 648, 649, 650,
                        654, 656, 657, 658, 660, 661, 663, 664, 665, 666, 670, 672, 673, 674, 678, 683]
    active_carrier_S1_2 = [684, 689, 692, 696, 698, 699, 701, 702, 703, 704, 706, 707, 708,
                        712, 714, 715, 717, 718, 719, 720, 722, 723, 725, 726, 727, 729,
                        733, 734, 735, 736, 738, 739, 740, 744, 746, 747, 748, 753, 756,
                        760, 762, 763, 765, 766, 767, 768, 770, 771, 772, 776, 778, 779,
                        780, 785, 788, 792, 794, 795, 796, 801, 805, 806, 807, 809]

    KP1 = np.concatenate((active_carrier_S1_1, active_carrier_S2, active_carrier_S1_2), axis=0)
    # active_carrier_S1 >> length 64 / active_carrier_S2 >> length 256
    # KP1 is the active carrier's index and its fixed



    ##### 9.8.2.2 Modulation of the Active Carriers in P1
    # total length of each S1 pattern is 64
    S1_pattern_HEX = np.array([['124721741D482E7B'],    # 000
                                ['47127421481D7B2E'],   # 001
                                ['217412472E7B1D48'],   # 010
                                ['742147127B2E481D'],   # 011
                                ['1D482E7B12472174'],   # 100
                                ['481D7B2E47127421'],   # 101
                                ['2E7B1D4821741247'],   # 110
                                ['7B2E481D74214712']])  # 111
    S1_pattern_BIN = np.zeros([S1_pattern_HEX.size, 64])

    # total length of each S2 pattern is 256
    S2_pattern_HEX = np.array([['121D4748212E747B1D1248472E217B7412E247B721D174841DED48B82EDE7B8B'],    # 0000
                                ['4748121D747B212E48471D127B742E2147B712E2748421D148B81DED7B8B2EDE'],   # 0001
                                ['212E747B121D47482E217B741D12484721D1748412E247B72EDE7B8B1DED48B8'],   # 0010
                                ['747B212E4748121D7B742E2148471D12748421D147B712E27B8B2EDE48B81DED'],   # 0011
                                ['1D1248472E217B74121D4748212E747B1DED48B82EDE7B8B12E247B721D17484'],   # 0100
                                ['48471D127B742E214748121D747B212E48B81DED7B8B2EDE47B712E2748421D1'],   # 0101
                                ['2E217B741D124847212E747B121D47482EDE7B8B1DED48B821D1748412E247B7'],   # 0110
                                ['7B742E2148471D12747B212E4748121D7B8B2EDE48B81DED748421D147B712E2'],   # 0111
                                ['12E247B721D174841DED48B82EDE7B8B121D4748212E747B1D1248472E217B74'],   # 1000
                                ['47B712E2748421D148B81DED7B8B2EDE4748121D747B212E48471D127B742E21'],   # 1001
                                ['21D1748412E247B72EDE7B8B1DED48B8212E747B121D47482E217B741D124847'],   # 1010
                                ['748421D147B712E27B8B2EDE48B81DED747B212E4748121D7B742E2148471D12'],   # 1011
                                ['1DED48B82EDE7B8B12E247B721D174841D1248472E217B74121D4748212E747B'],   # 1100
                                ['48B81DED7B8B2EDE47B712E2748421D148471D127B742E214748121D747B212E'],   # 1101
                                ['2EDE7B8B1DED48B821D1748412E247B72E217B741D124847212E747B121D4748'],   # 1110
                                ['7B8B2EDE48B81DED748421D147B712E27B742E2148471D12747B212E4748121D']])  # 1111
    S2_pattern_BIN = np.zeros([S2_pattern_HEX.size, 256])

    for idx, pat in enumerate(S1_pattern_HEX):
        pat = list(pat[0])
        temp = []
        for val in pat:
            int_val = int(val, base=16)
            temp += format(int_val, 'b').zfill(4)
        S1_pattern_BIN[idx,:] = temp

    for idx, pat in enumerate(S2_pattern_HEX):
        pat = list(pat[0])
        temp = []
        for val in pat:
            int_val = int(val, base=16)
            temp += format(int_val, 'b').zfill(4)
        S2_pattern_BIN[idx,:] = temp


    CSS_S1 = S1_pattern_BIN[S1,:]
    CSS_S2 = S2_pattern_BIN[S2,:]

    MSS_SEQ = np.concatenate((CSS_S1, CSS_S2, CSS_S1), axis=0)
    MSS_DIFF = np.zeros(MSS_SEQ.size)

    # DBPSK
    temp = 1
    for idx, val in enumerate(MSS_SEQ):
        if val == 1: temp = -temp
        MSS_DIFF[idx] = temp

    # SCRAMBLING
    PRBS = np.zeros(384)
    PRBS_init = bin(0b100111001000110)
    PRBS_init = list(PRBS_init[2:])

    for i in range(384):
        next_bit = int(PRBS_init[-1])^int(PRBS_init[-2])  # 1+x^14+x^15
        PRBS_init.insert(0, next_bit)
        PRBS_init.pop(-1)
        PRBS[i] = next_bit

    SCR_SEQ = 1-2*PRBS

    MSS_SCR = MSS_DIFF*SCR_SEQ


    ##### 9.8.2.4 Generation of the time domain P1 symbol
    p1_len = 2048   # p1 symbol length
    p1C_len = 542
    p1A_len = 1024
    p1B_len = 482

    P1 = np.zeros(p1_len, dtype=complex)
    p1C = np.zeros(p1C_len, dtype=complex)
    p1A = np.zeros(p1A_len, dtype=complex)
    p1B = np.zeros(p1B_len, dtype=complex)

    # p1C generation
    for n in range(len(p1C)):
        temp = 0
        for i in range(384):
            temp += 1/np.sqrt(384)*MSS_SCR[i]*np.exp(2j*np.pi*n*(KP1[i]-426)/1024)
        p1C[n] = temp*np.exp(2j*np.pi*n/1024)

    # p1A generation
    for n in range(len(p1A)):
        temp = 0
        for i in range(384):
            temp += 1/np.sqrt(384)*MSS_SCR[i]*np.exp(2j*np.pi*n*(KP1[i]-426)/1024)
        p1A[n] = temp

    # p1B generation
    for n in range(len(p1B)):
        temp = 0
        for i in range(384):
            temp += 1/np.sqrt(384)*MSS_SCR[i]*np.exp(2j*np.pi*(n+542)*(KP1[i]-426)/1024)
        p1B[n] = temp*np.exp(2j*np.pi*(n+1566)/1024)

    P1 = np.concatenate((p1C, p1A, p1B), axis=0, dtype=complex)

    return P1
