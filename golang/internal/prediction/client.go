package prediction

import (
	"fmt"

	pb "github.com/automatedtomato/rice-yield-prediction/golang/internal/proto/rice_yield"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type PredictionClient struct {
	conn   *grpc.ClientConn
	client pb.YieldPredictionServiceClient
}

func NewPredictionClient(serverAddr string) (*PredictionClient, error) {
	conn, err := grpc.Dial(serverAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to gRPC server: %v", err)
	}

	client := pb.NewYieldPredictionServiceClient(conn)
	return &PredictionClient{
		conn:   conn,
		client: client,
	}, nil
}
