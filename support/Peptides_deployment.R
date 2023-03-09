boman_index <- boman(seq = current_sequence)  
weight <- mw(seq = current_sequence)
AAC <- aaComp(seq = current_sequence)[[1]]
charge <- charge(seq = current_sequence)
pI <- pI(seq = current_sequence, pKscale = "EMBOSS")
aindex <- aIndex(seq = current_sequence)
instability <- instaIndex(seq = current_sequence)
hydrophobicity <- hydrophobicity(seq = current_sequence)
hydrophobicity_moment <- hmoment(seq = current_sequence)
membrane_position <- membpos(seq = current_sequence)[[1]][1,]