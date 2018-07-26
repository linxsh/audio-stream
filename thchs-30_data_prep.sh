#!/bin/sh

for x in $1; do
	echo "cleaning $x"
	cd ./$x
	rm -rf wav.scp utt2spk spk2utt word.txt phone.txt text
	echo "preparing scps and text in $x"
	for nn in `find ./* -name '*'.wav|sort -u|xargs -i basename {} .wav`; do
		spkid=`echo $nn| awk -F"_" '{print"" $1}'`
		spk_char=`echo $spkid| sed 's/\([A−Z]*\).*/\1/'`
		spk_num=`echo $spkid| sed 's/\([A-Z]*\)\([0-9]\)/\2/'`
		spkid=$(printf '%s%.2d' "$spk_char" "$spk_num")
		utt_num=`echo $nn| awk -F"_" '{print $2}'`
		uttid=$(printf '%s%.2d_%.3d' "$spk_char" "$spk_num" "$utt_num")
		echo $uttid ./$x/$nn.wav >> wav.scp
		echo $uttid $spkid >> utt2spk
		echo $uttid `sed -n 1p ./$nn.wav.trn` >> word.txt
		echo $uttid `sed -n 3p ./$nn.wav.trn` >> phone.txt
	done
	cp word.txt text
	sort wav.scp -o wav.scp
	sort utt2spk -o utt2spk
	sort text -o text
	sort phone.txt -o phone.txt
done
