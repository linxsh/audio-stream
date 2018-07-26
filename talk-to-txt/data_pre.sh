#!/bin/sh

echo "create files from $1 to $2 .."
cur_dir=$(pwd)
src_dir=$cur_dir/$1
dst_dir=$cur_dir/$2/train
if [ ! -d $dst_dir ]; then
	mkdir $dst_dir
fi
rm $dst_dir/* -rf
cd $src_dir
for nn in `find ./* -name '*'.wav|sort -u|xargs -i basename {} .wav`; do
	spkid=`echo $nn | awk -F"_" '{print"" $1}'`
	spk_char=`echo $spkid | sed 's/\([A-Z]*\)\([0-9]*\).*/\1/'`
	spk_num=`echo $spkid | sed 's/\([A-Z]*\)\([0-9]*\).*/\2/'`
	spkid=$(printf '%s%.2d' "$spk_char" "$spk_num")
	utt_num=`echo $nn | awk -F"_" '{print $2}'`
	uttid=$(printf '%s%.2d_%.3d' "$spk_char" "$spk_num" "$utt_num")
	files=$(cat ./$nn.wav.trn)
	echo $uttid `sed -n 1p $src_dir/$files`>>$dst_dir/word.txt
	echo $uttid `sed -n 3p $src_dir/$files`>>$dst_dir/phone.txt
	cp $src_dir/$nn.wav $dst_dir/$uttid.wav -f
done
sort $dst_dir/word.txt  -o $dst_dir/word.txt
sort $dst_dir/phone.txt -o $dst_dir/phone.txt
