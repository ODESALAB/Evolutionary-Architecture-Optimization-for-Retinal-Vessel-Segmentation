import torch
import timeit
import torch.nn as nn
import torch.optim as optim
from cell_module.ops import OPS as ops_dict
from utils.early_stopping import EarlyStopping
from cell_module.encoder_cell import EncoderCell
from cell_module.decoder_cell import DecoderCell

class Model(nn.Module):
    def __init__(self, vector, config, node_boundaries):
        super(Model, self).__init__()

        self.NBR_HIDDEN_NODES = 5 # intermediate nodes + output node
        
        self.solNo = None
        self.fitness = -1
        self.cost = -1
        self.vector = vector
        self.config = config
        self.nbr_cell = int(config[-2])
        self.nbr_filters = int(config[-1])
        self.node_boundaries = node_boundaries

        self.cells = nn.ModuleList([])
        self.mp = nn.MaxPool2d((2, 2))
        self.compile()
            
        for param in self.parameters():
            param.requires_grad = True
            if len(param.shape) > 1:
                torch.nn.init.xavier_uniform_(param)

    def compile(self):
        """ Build UNAS-Net Model """
        C_in = 1
        C_out = 2**self.nbr_filters

        # Encoder Cell
        for cell_idx in range(self.nbr_cell):
            cell = EncoderCell(self.config, self.node_boundaries, C_in, C_out)
            C_in = C_out
            C_out = C_out * 2
            self.cells.append(cell)

        # Bottleneck
        self.cells.append(ops_dict['bottleneck'](C_in, C_out))

        # Decoder Cell
        for _ in range(self.nbr_cell):
            cell = DecoderCell(self.config, self.node_boundaries, C_in + C_in, C_in)
            self.cells.append(cell)
            C_in = C_in // 2
            C_out = C_out // 2
        
        # Output
        self.cells.append(nn.Conv2d(2**self.nbr_filters, 1, kernel_size=1, padding=0))

    def forward(self, inputs):
        
        outputs = [0] * (self.nbr_cell * 2 + 2) # Cell outputs
        pool_outputs = [0] * len(outputs) # Pooling outputs
        outputs[0] = inputs 

        # Encoder Cells
        cell_nbr = 1
        _input = inputs
        while cell_nbr <= self.nbr_cell:
            cell_output = self.cells[cell_nbr - 1](_input) # Feed forward input to Cell
            outputs[cell_nbr] = cell_output # Store output of Cell
            pool_outputs[cell_nbr] = self.mp(cell_output) # Apply Max Pooling
            _input = pool_outputs[cell_nbr] # Store output of Pooling operation

            cell_nbr += 1

        # Bottleneck
        outputs[cell_nbr] = self.cells[cell_nbr - 1](pool_outputs[cell_nbr - 1])
        cell_nbr += 1
        
        # Decoder Cells
        skip_idx = cell_nbr - 2
        while cell_nbr < len(self.cells):
            _input = outputs[cell_nbr - 1]
            cell_output = self.cells[cell_nbr - 1](_input, outputs[skip_idx])
            outputs[cell_nbr] = cell_output 

            skip_idx -= 1
            cell_nbr += 1
        
        # Output
        outputs.append(self.cells[-1](outputs[-1]))

        return outputs[-1]
    
    def evaluate(self, train_loader, val_loader, loss_fn, metric_fn, device):
        
        try:
            print(f"Model {self.solNo} Training...")
            self.to(device) # cuda start

            train_loss = []
            train_dice = []
            log = f"Model No: {self.solNo}\n"
            early_stopping = EarlyStopping(patience=10)

            startTime = timeit.default_timer()
            optimizer = optim.Adam(self.parameters(), lr=0.0001)
            for epoch in range(120):

                # Train Phase
                self.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
        
                    with torch.set_grad_enabled(True):
                        output = self.forward(inputs)
                        error = loss_fn(output, labels)
                        train_loss.append(error.item())
                        train_dice.append(metric_fn(output, labels).item())
                        optimizer.zero_grad()
                        error.backward()
                        optimizer.step()
                
                torch.cuda.empty_cache()
		
                # Validation Phase
                val_loss = []
                val_dice = []
                self.eval()
                #with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = self.forward(inputs)
                    error = loss_fn(output, labels)
                    val_dice.append(metric_fn(output, labels).item())
                    val_loss.append(error.item())
                
                torch.cuda.empty_cache()
                
                # Log
                avg_tr_loss = sum(train_loss) / len(train_loss)
                avg_tr_score = sum(train_dice) / len(train_dice)
                avg_val_loss = sum(val_loss) / len(val_loss)
                avg_val_score = sum(val_dice) / len(val_dice)
                txt = f"\nEpoch: {epoch}, tr_loss: {avg_tr_loss}, tr_dice_score: {avg_tr_score}, val_loss: {avg_val_loss}, val_dice: {avg_val_score}"
                log += txt
                print(txt)

                # Early Stopping Check
                if early_stopping.stopTraining(epoch, avg_val_loss, avg_val_score):
                    self.fitness = early_stopping.best_score
                    self.cost = timeit.default_timer() - startTime
                    print(f"Stop Training - Model {self.solNo} , {self.fitness}, {self.cost}")
                    break
            
        except Exception as e: # Memory Problems
            torch.cuda.empty_cache()
            print(e)
            return -1, -1, None

        torch.cuda.empty_cache()

        self.fitness = early_stopping.best_score
        self.cost = timeit.default_timer() - startTime
        
        log += f"\nElapsed Time: {self.cost}, Fitness: {self.fitness}"
        
        return self.fitness, self.cost, log
    
    def reset(self):
        for param in self.parameters():
            param.requires_grad = True
            if len(param.shape) > 1:
                torch.nn.init.xavier_uniform_(param)
                param.data.grad = None