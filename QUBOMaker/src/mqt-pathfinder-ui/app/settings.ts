class Settings {

    encoding: number = -1;
    nPaths: number = 1;
    maxPathLength: number = 4;

    loop: boolean = false;
    minimizeWeight: boolean = false;
    maximizeWeight: boolean = false;

    exactlyOnceVertices: number[] = [];
    atLeastOnceVertices: number[] = [];
    atMostOnceVertices: number[] = [];

    exactlyOnceEdges: string[][] = [];
    atLeastOnceEdges: string[][] = [];
    atMostOnceEdges: string[][] = [];

    precedences: string[][] = [];

    shareNoEdges: number[] = [];
    shareNoVertices: number[] = [];

    setEncoding(encoding: boolean[]) {
        const clone = this.clone();
        clone.encoding = encoding.indexOf(true);
        return clone;
    }

    setNPaths(nPaths: number) {
        const clone = this.clone();
        clone.nPaths = nPaths;
        return clone;
    }

    setMaxPathLength(maxPathLength: number) {
        const clone = this.clone();
        clone.maxPathLength = maxPathLength;
        return clone;
    }

    setGeneralSettings(settings: boolean[]) {
        const clone = this.clone();
        clone.loop = settings[0];
        clone.minimizeWeight = settings[1];
        clone.maximizeWeight = settings[2];
        return clone;
    }

    setExactlyOnceVertices(exactlyOnceVertices: boolean[]) {
        const clone = this.clone();
        clone.exactlyOnceVertices = exactlyOnceVertices.slice(1).map((value, index) => value ? index : -1).filter(value => value !== -1);
        return clone;
    }

    setAtLeastOnceVertices(atLeastOnceVertices: boolean[]) {
        const clone = this.clone();
        clone.atLeastOnceVertices = atLeastOnceVertices.slice(1).map((value, index) => value ? index : -1).filter(value => value !== -1);
        return clone;
    }

    setAtMostOnceVertices(atMostOnceVertices: boolean[]) {
        const clone = this.clone();
        clone.atMostOnceVertices = atMostOnceVertices.slice(1).map((value, index) => value ? index : -1).filter(value => value !== -1);
        return clone;
    }

    setShareNoVertices(shareNoVertices: boolean[]) {
        const clone = this.clone();
        clone.shareNoVertices = shareNoVertices.slice(1).map((value, index) => value ? index : -1).filter(value => value !== -1);
        return clone;
    }

    setShareNoEdges(shareNoEdges: boolean[]) {
        const clone = this.clone();
        clone.shareNoEdges = shareNoEdges.slice(1).map((value, index) => value ? index : -1).filter(value => value !== -1);
        return clone;
    }

    setExactlyOnceEdges(exactlyOnceEdges: string[][]) {
        const clone = this.clone();
        clone.exactlyOnceEdges = exactlyOnceEdges;
        return clone;
    }

    setAtLeastOnceEdges(atLeastOnceEdges: string[][]) {
        const clone = this.clone();
        clone.atLeastOnceEdges = atLeastOnceEdges;
        return clone;
    }

    setAtMostOnceEdges(atMostOnceEdges: string[][]) {
        const clone = this.clone();
        clone.atMostOnceEdges = atMostOnceEdges;
        return clone;
    }

    setPrecedence(precedences: string[][]) {
        const clone = this.clone();
        clone.precedences = precedences;
        return clone;
    }

    clone() {
        let clone = new Settings();
        clone.encoding = this.encoding;
        clone.nPaths = this.nPaths;
        clone.maxPathLength = this.maxPathLength;
        clone.loop = this.loop;
        clone.minimizeWeight = this.minimizeWeight;
        clone.maximizeWeight = this.maximizeWeight;
        clone.exactlyOnceVertices = this.exactlyOnceVertices.slice();
        clone.atLeastOnceVertices = this.atLeastOnceVertices.slice();
        clone.atMostOnceVertices = this.atMostOnceVertices.slice();
        clone.exactlyOnceEdges = this.exactlyOnceEdges.map(value => value.slice());
        clone.atLeastOnceEdges = this.atLeastOnceEdges.map(value => value.slice());
        clone.atMostOnceEdges = this.atMostOnceEdges.map(value => value.slice());
        clone.precedences = this.precedences.map(value => value.slice());
        clone.shareNoEdges = this.shareNoEdges.slice();
        clone.shareNoVertices = this.shareNoVertices.slice();
        return clone;
    }

    toJson() {
        if(this.encoding === -1) {
            throw new Error("Encoding not set");
        }
        const object: { [key: string]: any} = {};
        object["settings"] = {
            "encoding": this.encoding,
            "n_paths": this.nPaths,
            "max_path_length": this.maxPathLength,
            "loop": this.loop,
        };
        if(this.minimizeWeight && this.maximizeWeight) {
            throw new Error("Cannot minimize and maximize weight at the same time");
        } else if(this.minimizeWeight) {
            object["objective_function"] = {
                "type": "MinimisePathLength",
                "path_ids": Array(this.nPaths).fill(0).map((_, index) => index + 1)
            };
        } else if(this.maximizeWeight) {
            object["objective_function"] = {
                "type": "MaximisePathLength",
                "path_ids": Array(this.nPaths).fill(0).map((_, index) => index + 1)
            };
        }

        const constraints: { [key: string]: any}[] = [];
        constraints.push({
            "type": "PathIsValid",
            "path_ids": Array(this.nPaths).fill(0).map((_, index) => index + 1),
        });
        //--------------------- Vertices ---------------------
        if(this.exactlyOnceVertices.length > 0) {
            constraints.push({
               "type": "PathContainsVerticesExactlyOnce",
               "path_ids": Array(this.nPaths).fill(0).map((_, index) => index + 1),
               "vertices": this.exactlyOnceVertices
            });
        }
        if(this.atLeastOnceVertices.length > 0) {
            constraints.push({
               "type": "PathContainsVerticesAtLeastOnce",
               "path_ids": Array(this.nPaths).fill(0).map((_, index) => index + 1),
               "vertices": this.atLeastOnceVertices
            });
        }
        if(this.atMostOnceVertices.length > 0) {
            constraints.push({
               "type": "PathContainsVerticesAtMostOnce",
               "path_ids": Array(this.nPaths).fill(0).map((_, index) => index + 1),
               "vertices": this.atMostOnceVertices
            });
        }
        //--------------------- Edges ---------------------
        if(this.exactlyOnceEdges.length > 0) {
            constraints.push({
               "type": "PathContainsEdgesExactlyOnce",
               "path_ids": Array(this.nPaths).fill(0).map((_, index) => index + 1),
               "edges": this.exactlyOnceEdges
            });
        }
        if(this.atLeastOnceEdges.length > 0) {
            constraints.push({
               "type": "PathContainsEdgesAtLeastOnce",
               "path_ids": Array(this.nPaths).fill(0).map((_, index) => index + 1),
               "edges": this.atLeastOnceEdges
            });
        }
        if(this.atMostOnceEdges.length > 0) {
            constraints.push({
               "type": "PathContainsEdgesAtMostOnce",
               "path_ids": Array(this.nPaths).fill(0).map((_, index) => index + 1),
               "edges": this.atMostOnceEdges
            });
        }
        if(this.precedences.length > 0) {
            constraints.push({
               "type": "PrecedenceConstraint",
               "path_ids": Array(this.nPaths).fill(0).map((_, index) => index + 1),
               "precedences": this.precedences.map(value => ({
                    "before": value[0],
                    "after": value[1]
               }))
            });
        }

        if(this.shareNoEdges.length > 0) {
            constraints.push({
                "type": "PathsShareNoEdges",
                "path_ids": Array(this.nPaths).fill(0).map((_, index) => index + 1),
            });
        }
        if(this.shareNoVertices.length > 0) {
            constraints.push({
                "type": "PathsShareNoVertices",
                "path_ids": Array(this.nPaths).fill(0).map((_, index) => index + 1),
            });
        }

        object["constraints"] = constraints;
        return JSON.stringify(object);
    }
}

export default Settings
