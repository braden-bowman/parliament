"""
Contributors:
    Braden Bowman
        bradenwbowman@gmail.com
        
Updated:
    2024-03-16
    
to simplify this..... rather than several one to ones
    1. convert to polars
    2. convert to anything else
"""




class GraphConverter():
    """
    takes a graph from a python graphing library and stores it in a polars.DataFrame
    """
    
    def __init__(self, graph, out_type=None, in_type=None):
        
        self.graph = graph
    
        if out_type==None:
            self.out_type = polars.DataFrame
        else:
            self.out_type = out_type
            
        if in_type==None:
            self.in_type = type(graph)
        else:
            self.in_type = in_type

    
    def graph_to_polars():
        # basically, take all attributes of the graph and put them in polars
        pass
    
    def to_networkx():
        # each to f(x) should check for each of its own attributes and try to pull them from polars
        
        pass
    
    def to_cugraph():
        pass
    
    def to_rustworkx():
        pass
    
    def to_pyvis():
        pass
    
    def to_holoviews(): 
        pass
    
    
    
    

