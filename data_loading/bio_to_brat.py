from os.path import join
import argparse


def create_brat_from_bio(bio, output_folder):

    output = None
    ann = None

    with open(bio) as file:
        offset = 0
        document_idx = 0
        a_idx = 0
        within_annotation = False
        tokens = ''
        for line in file:
            line = line.rstrip()
            if line.startswith("-DOCSTART-"):
                # new document
                s = line.split(':')
                if len(s) == 2:
                    filename = s[1]
                else:
                    filename = 'document' + str(document_idx)

                document_idx += 1
                offset = 0
                a_idx = 1
                if (document_idx > 1):
                    output.close()
                    ann.close()
                output = open(join(output_folder,filename + '.txt'),'w')
                ann = open(join(output_folder,filename + '.ann'),'w')
                continue
            elif line == '':
                # line empty, next sentence
                output.write('\n')
                offset += 1
                continue
            token, annotation = line.split("\t")
            if not annotation.startswith('I-'):
                # either a new software starts or a old software ends
                if within_annotation:
                    ann.write(str(offset - 1) + '\t' + tokens + '\n')
                    #offset += 1
                    within_annotation = False
            if annotation.startswith('B-'):
                # new annotation tag
                atag = annotation.split('-')[1]
                ann.write("T" + str(a_idx) +"\t" + str(atag) +' ' + str(offset) + ' ')
                tokens = token
                #ann.write("T" + str(a_idx) +"\t" + str(annotation) +' ' + str(offset) + ' ' + str(offset+len(token)) + "\t" + str(token))
                a_idx +=1
                within_annotation = True
            elif annotation.startswith('I-'):
                tokens = tokens + ' ' + token
                            
            offset += len(token) + 1
            output.write(token + " ")
        output.close()
        ann.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Converts a BIO representation to BRAT annotation format")

    parser.add_argument("--input-file", required=True, help="Path to the BIO annotation file")
    parser.add_argument("--output-folder", required=True, help="Path to folder for BRAT output")
    args = parser.parse_args()

    print(args)

    create_brat_from_bio(bio=args.input_file, output_folder=args.output_folder)