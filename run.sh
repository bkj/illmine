
PROJECT_ROOT="$(pwd)"
cd $PROJECT_ROOT

# --
# Compiling code

mkdir bin
git clone https://github.com/jgrapht/jgrapht
cd jgrapht
mvn install
mvn package
cd ..

wget https://math.nist.gov/javanumerics/jama/Jama-1.0.3.zip
unzip Jama-1.0.3.zip
rm Jama-1.0.3.zip

javac -cp jgrapht/jgrapht-core/target/classes/:. -d bin */*java 


./compile.sh

# --
# Make data

mkdir -p data/synthetic/
mkdir -p data/synthetic/results/

wget http://www.cse.psu.edu/~madduri/software/GTgraph/GTgraph.tar.gz
tar -zxvf GTgraph.tar.gz
cd GTgraph
# Change CC=gcc in Makefile.var
make 
make
make rmat
cp R-MAT/GTgraph-rmat ../
cd ..
rm -rf GTgraph.tar.gz GTgraph


mkdir graphs
cd graphs

# make graph
$PROJECT_ROOT/GTgraph-rmat -n 1000 -m 12000 -o ./sample_small.txt
rm log

# convert format
java -cp $PROJECT_ROOT/bin -Xmx20g dataGen/GTGraphToOurFormatConverter \
    $PROJECT_ROOT/data/synthetic/graphs sample_small.txt GT_small.txt types_small.txt

# preprocessing
time java -cp $PROJECT_ROOT/bin -Xmx5g IndexConstruction/SortedEdgeListsConstructor \
    $PROJECT_ROOT/data/synthetic/graphs GT_small.txt types_small.txt

time java -cp $PROJECT_ROOT/bin -Xmx5g IndexConstruction/SPDAndTopologyAndSPathIndexConstructor2 \
    $PROJECT_ROOT/data/synthetic/graphs GT_small.txt types_small.txt 2

cd ..

# make queries
mkdir -p queries/
for i in `seq 2 5`; do
    for j in `seq 1 5`; do
        for type in Path Clique Subgraph; do
            echo "Generating query for ID: ${j} and #Nodes: ${i} and Type: ${type}";
            java -Xmx10g -cp $PROJECT_ROOT/bin QueryExecution/Random${type}QueryGenerator $PROJECT_ROOT/data/synthetic/queries/ dummy ../graphs/types_small.txt ${i}
            mv $PROJECT_ROOT/data/synthetic/queries/queryGraph.txt $PROJECT_ROOT/data/synthetic/queries/queryGraph.${type}.${j}.${i}.txt  
            mv $PROJECT_ROOT/data/synthetic/queries/queryTypes.txt $PROJECT_ROOT/data/synthetic/queries/queryTypes.${type}.${j}.${i}.txt
        done
    done 
done

java -Xmx10g -cp $PROJECT_ROOT/bin:$PROJECT_ROOT/jgrapht/jgrapht-core/target/classes QueryExecution/QBSQueryExecutorV2 \
    $PROJECT_ROOT/data/synthetic \
    graphs/GT_small.txt \
    graphs/types_small.txt \
    2 \
    queries/queryGraph.Path.1.4.txt \
    queries/queryTypes.Path.2.4.txt \
    graphs/GT_small.spath \
    50 \
    graphs/GT_small.topology \
    graphs/GT_small.spd \
    results/




